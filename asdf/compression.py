# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

import struct

import numpy as np


DEFAULT_BLOCK_SIZE = 1 << 22  #: Decompressed block size in bytes, 4MiB


def validate(compression):
    """
    Validate the compression string.

    Parameters
    ----------
    compression : str, bytes or None

    Returns
    -------
    compression : str or None
        In canonical form.

    Raises
    ------
    ValueError
    """
    if not compression or compression == b'\0\0\0\0':
        return None

    if isinstance(compression, bytes):
        compression = compression.decode('ascii')

    compression = compression.strip('\0')
    if compression not in ('zlib', 'bzp2', 'lz4', 'blsc', 'input'):
        raise ValueError(
            "Supported compression types are: 'zlib', 'bzp2', 'lz4', 'blsc', or 'input'")

    return compression


class Lz4Compressor:
    def __init__(self, block_api):
        self._api = block_api

    def compress(self, data):
        output = self._api.compress(data, mode='high_compression')
        header = struct.pack('!I', len(output))
        return header + output


class Lz4Decompressor:
    def __init__(self, block_api):
        self._api = block_api
        self._size = 0
        self._pos = 0
        self._buffer = b''

    def decompress(self, data):
        if not self._size:
            data = self._buffer + data
            if len(data) < 4:
                self._buffer += data
                return b''
            self._size = struct.unpack('!I', data[:4])[0]
            data = data[4:]
            self._buffer = bytearray(self._size)
        if self._pos + len(data) < self._size:
            self._buffer[self._pos:self._pos + len(data)] = data
            self._pos += len(data)
            return b''
        else:
            offset = self._size - self._pos
            self._buffer[self._pos:] = data[:offset]
            data = data[offset:]
            self._size = 0
            self._pos = 0
            output = self._api.decompress(self._buffer)
            self._buffer = b''
            return output + self.decompress(data)


class BloscCompressor:
    def __init__(self, blosc, typesize=1, clevel=1, shuffle='shuffle', cname='zstd', nthreads=1, blocksize=512*1024):
        self.blosc = blosc
        self.typesize = typesize  # dtype size in bytes, e.g. 8 for int64
        self.clevel = clevel  # compression level, usually only need lowest for zstd
        self.cname = cname  # compressor name, default zstd, good performance/compression tradeoff
        if shuffle == 'shuffle':
            self.shuffle = blosc.SHUFFLE
        elif shuffle == 'bitshuffle':
            self.shuffle = blosc.BITSHUFFLE
        else:
            self.shuffle = blosc.NOSHUFFLE

        # These could someday be user-configurable
        blosc.set_nthreads(nthreads)
        blosc.set_blocksize(blocksize)

        #print(f'blosc configured with typesize {typesize}, shuffle {shuffle}, blocksize {blocksize/(1<<20):.2f} MB')


    def compress(self, data):
        if data.nbytes > 2147483631:  # ~2 GB
            # This should never happen, because we compress in blocks that are 4 MiB
            raise ValueError("data blocks must be smaller than 2147483631 bytes due to internal blosc limitations")
        if self.typesize == 'auto':
            this_typesize = data.itemsize
        else:
            this_typesize = self.typesize
        assert this_typesize != 1
        compressed = self.blosc.compress(data, typesize=this_typesize, clevel=self.clevel, shuffle=self.shuffle, cname=self.cname)
        header = struct.pack('!I', len(compressed))
        # TODO: this probably triggers a data copy, feels inefficient. Probably have to add output array arg to blosc to fix
        return header + compressed  # bytes type

import time
class BloscDecompressor:
    tottime = 0.
    def __init__(self, blosc, nthreads=1):
        self.blosc = blosc
        self._size = 0
        self._pos = 0
        self._buffer = None
        self._partial_len = b''

        blosc.set_nthreads(nthreads)

    def decompress_into(self, data, out):
        bytesout = 0
        while len(data):
            if not self._size:
                # Don't know the (compressed) length of this block yet
                if len(self._partial_len) + len(data) < 4:
                    self._partial_len += data
                    break  # we've exhausted the data
                if self._partial_len:
                    # If we started to fill a len key, finish filling it
                    remaining = 4-len(self._partial_len)
                    if remaining:
                        self._partial_len += data[:remaining]
                        data = data[remaining:]
                    self._size = struct.unpack('!I', self._partial_len)[0]
                    self._partial_len = b''
                else:
                    # Otherwise just read the len key directly
                    self._size = struct.unpack('!I', data[:4])[0]
                    data = data[4:]
                
            if len(data) < self._size or self._buffer is not None:
                # If we have a partial block, or we're already filling a buffer, use the buffer
                if self._buffer is None:
                    self._buffer = np.empty(self._size, dtype=np.byte)  # use numpy instead of bytearray so we can avoid zero initialization
                    self._pos = 0
                newbytes = min(self._size - self._pos, len(data))  # don't fill past the buffer len!
                self._buffer[self._pos:self._pos+newbytes] = np.frombuffer(data[:newbytes], dtype=np.byte)
                self._pos += newbytes
                data = data[newbytes:]
            
                if self._pos == self._size:
                    start = time.perf_counter()
                    thisout = self.blosc.decompress(self._buffer)  # TODO: could avoid copy if we change the blosc API to accept an output destination
                    BloscDecompressor.tottime += time.perf_counter() - start
                    out[bytesout:bytesout+len(thisout)] = np.frombuffer(thisout, dtype=np.byte)
                    bytesout += len(thisout)
                    self._buffer = None
                    self._size = 0
            else:
                # We have at least one full block
                start = time.perf_counter()
                thisout = self.blosc.decompress(data[:self._size])
                BloscDecompressor.tottime += time.perf_counter() - start
                out[bytesout:bytesout+len(thisout)] = np.frombuffer(thisout, dtype=np.byte)
                bytesout += len(thisout)
                data = data[self._size:]
                self._size = 0

        return bytesout


def _get_decoder(compression, **kwargs):
    if compression == 'zlib':
        try:
            import zlib
        except ImportError:
            raise ImportError(
                "Your Python does not have the zlib library, "
                "therefore the compressed block in this ASDF file "
                "can not be decompressed.")
        return zlib.decompressobj()
    elif compression == 'bzp2':
        try:
            import bz2
        except ImportError:
            raise ImportError(
                "Your Python does not have the bz2 library, "
                "therefore the compressed block in this ASDF file "
                "can not be decompressed.")
        return bz2.BZ2Decompressor()
    elif compression == 'lz4':
        try:
            import lz4.block
        except ImportError:
            raise ImportError(
                "lz4 library in not installed in your Python environment, "
                "therefore the compressed block in this ASDF file "
                "can not be decompressed.")
        return Lz4Decompressor(lz4.block)
    elif compression == 'blsc':
        try:
            import blosc
        except ImportError:
            raise ImportError(
                'blosc library not installed in your Python environment, '
                'therefore the compressed block in this ASDF file '
                'can not be decompressed.  Install with: "pip install python-blosc"')
        return BloscDecompressor(blosc, **kwargs)
    else:
        raise ValueError(
            "Unknown compression type: '{0}'".format(compression))


def _get_encoder(compression, **kwargs):
    '''
    `compression` is the name of the compression,
    `typesize` is the size in bytes of the data type.  This information is used
    to increase the effectiveness of the compression.  Presently only used for `blosc`.
    '''
    if compression == 'zlib':
        try:
            import zlib
        except ImportError:
            raise ImportError(
                "Your Python does not have the zlib library, "
                "therefore the block in this ASDF file "
                "can not be compressed.")
        return zlib.compressobj()
    elif compression == 'bzp2':
        try:
            import bz2
        except ImportError:
            raise ImportError(
                "Your Python does not have the bz2 library, "
                "therefore the block in this ASDF file "
                "can not be compressed.")
        return bz2.BZ2Compressor()
    elif compression == 'lz4':
        try:
            import lz4.block
        except ImportError:
            raise ImportError(
                "lz4 library in not installed in your Python environment, "
                "therefore the block in this ASDF file "
                "can not be compressed.")
        return Lz4Compressor(lz4.block)
    elif compression == 'blsc':
        try:
            import blosc
        except ImportError:
            raise ImportError(
                "blosc library not installed in your Python environment, "
                "therefore the block in this ASDF file "
                'can not be compressed.  Install with: "pip install python-blosc"')
        return BloscCompressor(blosc, **kwargs)
    else:
        raise ValueError(
            "Unknown compression type: '{0}'".format(compression))


def to_compression_header(compression):
    """
    Converts a compression string to the four byte field in a block
    header.
    """
    if not compression:
        return b''

    if isinstance(compression, str):
        return compression.encode('ascii')

    return compression


def decompress(fd, used_size, data_size, compression):
    """
    Decompress binary data in a file

    Parameters
    ----------
    fd : generic_io.GenericIO object
         The file to read the compressed data from.

    used_size : int
         The size of the compressed data

    data_size : int
         The size of the uncompressed data

    compression : str
         The compression type used.

    Returns
    -------
    array : numpy.array
         A flat uint8 containing the decompressed data.
    """
    buffer = np.empty((data_size,), np.uint8)

    compression = validate(compression)
    decoder = _get_decoder(compression, **global_decompression_options)

    i = 0
    for block in fd.read_blocks(used_size):
        if hasattr(decoder, 'decompress_into'):
            i += decoder.decompress_into(block, out=buffer[i:])
        else:
            decoded = decoder.decompress(block)
            if i + len(decoded) > data_size:
                raise ValueError("Decompressed data too long")
            buffer.data[i:i+len(decoded)] = decoded
            i += len(decoded)

    if hasattr(decoder, 'flush'):
        decoded = decoder.flush()
        if i + len(decoded) > data_size:
            raise ValueError("Decompressed data too long")
        buffer[i:i+len(decoded)] = decoded
        i += len(decoded)
    
    if hasattr(decoder, '_buffer'):
        assert decoder._buffer is None
    if i != data_size:
        raise ValueError("Decompressed data wrong size")
    #print(BloscDecompressor.tottime)

    return buffer


global_compression_options = {}
def set_compression_options(**kwargs):
    global global_compression_options
    global_compression_options = kwargs.copy()

global_decompression_options = {}
def set_decompression_options(**kwargs):
    global global_decompression_options
    global_decompression_options = kwargs.copy()


def compress(fd, data, compression, block_size=DEFAULT_BLOCK_SIZE):
    """
    Compress array data and write to a file.

    Parameters
    ----------
    fd : generic_io.GenericIO object
        The file to write to.

    data : buffer
        The buffer of uncompressed data.

    compression : str
        The type of compression to use.

    block_size : int, optional
        Input data will be split into blocks of this size (in bytes) before compression.
    """
    compression = validate(compression)
    #if type(data) is np.memmap:
    #    raise NotImplementedError("memmap doesn't know about itemsize!")  # TODO
    #print(type(data), data.itemsize)
    #if data.itemsize == 1:
    #    raise ValueError("itemsize detection failed")  # TODO: propagate this information down in all cases!

    #typesize = data.itemsize
    #shuffle = 'bitshuffle'
    #if typesize == 8:
    #    shuffle = 'bitshuffle'
    #print(f'using typesize {typesize}, shuffle {shuffle}')

    block_size = global_compression_options.pop('asdf_block_size', block_size)
    encoder = _get_encoder(compression, **global_compression_options)

    # We can have numpy arrays here. While compress() will work with them,
    # it is impossible to split them into fixed size blocks without converting
    # them to bytes.
    if isinstance(data, np.ndarray):
        #data = data.tobytes()
        data = memoryview(data.reshape(-1))  # TODO: is it okay to use a view instead of a copy here?

    nelem = block_size // data.itemsize
    for i in range(0, len(data), nelem):
        fd.write(encoder.compress(data[i:i+nelem]))
    if hasattr(encoder, "flush"):
        fd.write(encoder.flush())


def get_compressed_size(data, compression, block_size=DEFAULT_BLOCK_SIZE):
    """
    Returns the number of bytes required when the given data is
    compressed.

    Parameters
    ----------
    data : buffer

    compression : str
        The type of compression to use.

    block_size : int, optional
        Input data will be split into blocks of this size (in bytes) before the compression.

    Returns
    -------
    bytes : int
    """
    compression = validate(compression)
    encoder = _get_encoder(compression, typesize=data.itemsize)

    l = 0
    for i in range(0, len(data), block_size):
        l += len(encoder.compress(data[i:i+block_size]))
    if hasattr(encoder, "flush"):
        l += len(encoder.flush())

    return l
