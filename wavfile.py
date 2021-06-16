from __future__ import division, print_function, absolute_import

import numpy
import struct
import warnings
import collections

class WavFileWarning(UserWarning):
    pass


_ieee = False




def _read_fmt_chunk(fid):
    res = struct.unpack('<ihHIIHH', fid.read(20))
    size, comp, noc, rate, sbytes, ba, bits = res
    if (comp != 1 or size > 16):
        if (comp == 3):
            global _ieee
            _ieee = True
            # Verificar se o audio é suportado
        else:
            warnings.warn("bytes desconhecidos", WavFileWarning)
        if (size > 16):
            fid.read(size - 16)
    return size, comp, noc, rate, sbytes, ba, bits



def _read_data_chunk(fid, noc, bits, normalized=False):
    size = struct.unpack('<i', fid.read(4))[0]

    if bits == 8 or bits == 24:
        dtype = 'u1'
        bytes = 1
    else:
        bytes = bits // 8
        dtype = '<i%d' % bytes

    if bits == 32 and _ieee:
        dtype = 'float32'

    data = numpy.fromfile(fid, dtype=dtype, count=size // bytes)

    if bits == 24:
        a = numpy.empty((len(data) // 3, 4), dtype='u1')
        a[:, :3] = data.reshape((-1, 3))
        a[:, 3:] = (a[:, 3 - 1:3] >> 7) * 255
        data = a.view('<i4').reshape(a.shape[:-1])

    if noc > 1:
        data = data.reshape(-1, noc)

    if bool(size & 1):
        fid.seek(1, 1)

    if normalized:
        if bits == 8 or bits == 16 or bits == 24:
            normfactor = 2 ** (bits - 1)
        data = numpy.float32(data) * 1.0 / normfactor

    return data


def _skip_unknown_chunk(fid): #pular pedaço desconhecido
    data = fid.read(4)
    if len(data) == 0:
        return
    size = struct.unpack('<i', data)[0]
    if bool(size & 1):
        size += 1
    fid.seek(size, 1)


def _read_riff_chunk(fid):
    str1 = fid.read(4)
    if str1 != b'RIFF':
        raise ValueError("Not a WAV file.")
    fsize = struct.unpack('<I', fid.read(4))[0] + 8
    str2 = fid.read(4)
    if (str2 != b'WAVE'):
        raise ValueError("Não é um arquivo WAV")
    return fsize


def read(file, readmarkers=False, readmarkerlabels=False, readmarkerslist=False, readloops=False, readpitch=False,
         normalized=False, forcestereo=False, log=False):
    # Retorna a taxa de amostragem do sinal
    if hasattr(file, 'read'):
        fid = file
    else:
        fid = open(file, 'rb')

    fsize = _read_riff_chunk(fid)
    noc = 1
    bits = 8

    _markersdict = collections.defaultdict(lambda: {'position': -1, 'label': ''})
    loops = []
    pitch = 0.0
    while (fid.tell() < fsize):

        chunk_id = fid.read(4)
        if chunk_id == b'fmt ':
            size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid)
        elif chunk_id == b'data':
            data = _read_data_chunk(fid, noc, bits, normalized)
        elif chunk_id == b'cue ':
            str1 = fid.read(8)
            size, numcue = struct.unpack('<ii', str1)
            for c in range(numcue):
                str1 = fid.read(24)
                id, position, datachunkid, chunkstart, blockstart, sampleoffset = struct.unpack('<iiiiii', str1)
                _markersdict[id]['position'] = position

        elif chunk_id == b'LIST':
            str1 = fid.read(8)
            size, type = struct.unpack('<ii', str1)
        elif chunk_id in [b'ICRD', b'IENG', b'ISFT', b'ISTJ']:
            _skip_unknown_chunk(fid)
        elif chunk_id == b'labl':
            str1 = fid.read(8)
            size, id = struct.unpack('<ii', str1)
            size = size + (size % 2)
            label = fid.read(size - 4).rstrip('\x00')  # remover caractres nulos

            _markersdict[id]['label'] = label

        elif chunk_id == b'smpl':
            str1 = fid.read(40)
            size, manuf, prod, sampleperiod, midiunitynote, midipitchfraction, smptefmt, smpteoffs, numsampleloops, samplerdata = struct.unpack(
                '<iiiiiIiiii', str1)
            cents = midipitchfraction * 1. / (2 ** 32 - 1)
            pitch = 440. * 2 ** ((midiunitynote + cents - 69.) / 12)
            for i in range(numsampleloops):
                str1 = fid.read(24)
                cuepointid, type, start, end, fraction, playcount = struct.unpack('<iiiiii', str1)
                loops.append([start, end])
        else:
            if log:
                warnings.warn("Chunk " + str(chunk_id) + " skipped", WavFileWarning)
            _skip_unknown_chunk(fid)
    fid.close()

    if data.ndim == 1 and forcestereo:
        data = numpy.column_stack((data, data))

    _markerslist = sorted([_markersdict[l] for l in _markersdict], key=lambda k: k['position'])  # Sortia posição
    _cue = [m['position'] for m in _markerslist]
    _cuelabels = [m['label'] for m in _markerslist]

    return (rate, data, bits,) \
           + ((_cue,) if readmarkers else ()) \
           + ((_cuelabels,) if readmarkerlabels else ()) \
           + ((_markerslist,) if readmarkerslist else ()) \
           + ((loops,) if readloops else ()) \
           + ((pitch,) if readpitch else ())


def write(filename, rate, data, bitrate=None, markers=None, loops=None, pitch=None, normalized=False):
    # Tratamento e normalização
    if bitrate == 24:
        if normalized:
            data[data > 1.0] = 1.0
            data[data < -1.0] = -1.0
            a32 = numpy.asarray(data * (2 ** 23 - 1), dtype=numpy.int32)
        else:
            a32 = numpy.asarray(data, dtype=numpy.int32)
        if a32.ndim == 1:
            a32.shape = a32.shape + (1,)  # Converta para uma matriz 2D com uma única coluna.
        a8 = (a32.reshape(a32.shape + (1,)) >> numpy.array([0, 8,
                                                            16])) & 255  # deslocar os primeiros 0 bits, depois 8, depois 16, a saída resultante é little-endian de 24 bits.
        data = a8.astype(numpy.uint8)
    else:
        if normalized:  # default to 32 bit int
            data[data > 1.0] = 1.0
            data[data < -1.0] = -1.0
            data = numpy.asarray(data * (2 ** 31 - 1), dtype=numpy.int32)

    fid = open(filename, 'wb')
    fid.write(b'RIFF')
    fid.write(b'\x00\x00\x00\x00')
    fid.write(b'WAVE')

    # fmt chunk
    fid.write(b'fmt ')
    if data.ndim == 1:
        noc = 1
    else:
        noc = data.shape[1]
    bits = data.dtype.itemsize * 8 if bitrate != 24 else 24
    sbytes = rate * (bits // 8) * noc
    ba = noc * (bits // 8)
    fid.write(struct.pack('<ihHIIHH', 16, 1, noc, rate, sbytes, ba, bits))

    # cue chunk
    if markers:
        if isinstance(markers[0], dict):
            labels = [m['label'] for m in markers]
            markers = [m['position'] for m in markers]
        else:
            labels = ['' for m in markers]

        fid.write(b'cue ')
        size = 4 + len(markers) * 24
        fid.write(struct.pack('<ii', size, len(markers)))
        for i, c in enumerate(markers):
            s = struct.pack('<iiiiii', i + 1, c, 1635017060, 0, 0, c)
            fid.write(s)

        lbls = ''
        for i, lbl in enumerate(labels):
            lbls += b'labl'
            label = lbl + ('\x00' if len(lbl) % 2 == 1 else '\x00\x00')
            size = len(lbl) + 1 + 4  # because \x00
            lbls += struct.pack('<ii', size, i + 1)
            lbls += label

        fid.write(b'LIST')
        size = len(lbls) + 4
        fid.write(struct.pack('<i', size))
        fid.write(
            b'adtl')
        fid.write(lbls)

        # smpl chunk
    if loops or pitch:
        if not loops:
            loops = []
        if pitch:
            midiunitynote = 12 * numpy.log2(pitch * 1.0 / 440.0) + 69
            midipitchfraction = int((midiunitynote - int(midiunitynote)) * (2 ** 32 - 1))
            midiunitynote = int(midiunitynote)

        else:
            midiunitynote = 0
            midipitchfraction = 0
        fid.write(b'smpl')
        size = 36 + len(loops) * 24
        sampleperiod = int(1000000000.0 / rate)

        fid.write(
            struct.pack('<iiiiiIiiii', size, 0, 0, sampleperiod, midiunitynote, midipitchfraction, 0, 0, len(loops), 0))
        for i, loop in enumerate(loops):
            fid.write(struct.pack('<iiiiii', 0, 0, loop[0], loop[1], 0, 0))


    fid.write(b'data')
    fid.write(struct.pack('<i', data.nbytes))
    import sys
    if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
        data = data.byteswap()

    data.tofile(fid)

    if data.nbytes % 2 == 1:
        fid.write('\x00')

    size = fid.tell()
    fid.seek(4)
    fid.write(struct.pack('<i', size - 8))
    fid.close()
