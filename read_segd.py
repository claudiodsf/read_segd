# -*- coding: utf8 -*-
"""
SEG D bindings to ObsPy core module.

:copyright:
    Claudio Satriano (satriano@ipgp.fr)
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str, PY2

from collections import OrderedDict
from struct import unpack
from array import array
import string
import math
import numpy as np
from obspy import UTCDateTime, Trace, Stream


class SEGDNotImplemented(Exception):
    pass


class SEGDScanTypeError(Exception):
    pass


# Code tables (SERCEL format)
_descale_multiplier = {
    0xAF6D: 1600,
    0xB76D: 400,
    0xAD03: 2500,
    0xB503: 650,
    # sanity value:
    0: 1
}
_record_types = {
    8: 'normal',
    2: 'test record'
}
_source_types = {
    0: 'no source',
    1: 'impulsive',
    2: 'vibro'
}
_test_record_types = {
    0: 'normal record',
    1: 'field (sensor) noise',
    2: 'field (sensor) tilt',
    3: 'field (sensor) crosstalk',
    4: 'instrument noise',
    5: 'instrument distortion',
    6: 'instrument gain/phase',
    7: 'instrument crosstalk',
    8: 'instrument common mode',
    9: 'synthetic',
    10: 'field (sensor) pulse',
    11: 'instrument pulse',
    12: 'field (sensor) distortion',
    13: 'instrument gravity',
    14: 'field (sensor) leakage',
    15: 'field (sensor) resistance'
}
_spread_types = {
    0: None,
    1: 'generic',
    2: 'absolute'
}
_noise_elimination_types = {
    1: 'off',
    2: 'diversity stack',
    3: 'historic',
    4: 'enhanced diversity stack'
}
_historic_editing_types = {
    1: 'zeroing',
    2: 'clipping'
}
_threshold_types = {
    0: None,
    1: 'hold',
    2: 'var'
}
_process_types = {
    1: 'no operation (raw data)',
    2: 'stack',
    3: 'correlation after stack',
    4: 'correlation before stack'
}
_filter_types = {
    1: 'minimum phase',
    2: 'linear phase'
}
_operating_modes = {
    0b10000: 'standard',
    0b01000: 'microseismic',
    0b00100: 'slip-sweep',
    0b00010: 'SQC dump (VSR)',
    0b00001: 'guidance (navigation)'
}
_dump_types = {
    0: 'normal dump',
    1: 'raw dump',
    2: 'extra dump'
}
_sensor_types = {
    0: 'not defined',
    1: 'hydrophone',
    2: 'geophone, vertical',
    3: 'geophone, horizontal, in-line',
    4: 'geophone, horizontal, crossline',
    5: 'geophone, horizontal, other',
    6: 'accelerometer, vertical',
    7: 'accelerometer, horizontal, in-line',
    8: 'accelerometer, horizontal, crossline',
    9: 'accelerometer, horizontal, other'
}
_unit_types = {
    0x00: 'not identified',
    0x01: 'FDU',
    0x03: 'RAU',
    0x1C: 'DSU',
    0x20: 'VE464'
}
_subunit_types = {
    0x01: 'FDU1-408',
    0x0F: 'FDU2S',
    0x15: 'FDU-428',
    0x16: 'DSU3-428',
    0x17: 'QT-428',
    0x1F: 'RAU 0x1E DSUGPS',
    0x21: 'DSU1-428, short',
    0x22: 'DSU3BV-428',
    0x24: 'DSU1-428, long',
    0x25: 'DSU3-SA',
    0x26: 'RAU-428',
    # sanity value:
    0: None
}
_channel_types = {
    0: 'geophone',
    1: 'hydrophone'
}
_control_unit_types = {
    0x30: 'LAUX-428',
    0x31: 'LCI-428',
    0x50: 'RAU',
    0x51: 'RAU-D',
    # sanity value:
    0: None,
    # other values found in files, but not documented:
    0x06: None
}
_channel_edited_statuses = {
    0: None,
    1: 'dead',
    2: 'acquisition/retrieve error',
    3: 'noise edition'
}
_channel_processes = {
    1: 'raw data',
    2: 'aux stack',
    3: 'correlation, negative part',
    4: 'correlation, positive part',
    5: 'normal correlation',
    6: 'seis stack'
}
_channel_gain_scales = {
    'FDU': (1600, 400),
    'RAU-428': (1600, 400),
    'DSU-428': (5, None),
    'DSU3-SA': (5, None),
    'RAU': (2500, 650)
}
_channel_gain_units = {
    'FDU': 'mV_RMS',
    'RAU-428': 'mV_RMS',
    'DSU-428': 'm/s/s',
    'DSU3-SA': 'm/s/s',
    'RAU': 'mV_peak'
}
_channel_filters = {
    'FDU': ('0.8FN minimum phase', '0.8FN linear phase'),
    'RAU-428': ('0.8FN minimum phase', '0.8FN linear phase'),
    'DSU-428': ('0.8FN minimum phase', '0.8FN linear phase'),
    'DSU3-SA': ('0.8FN minimum phase', '0.8FN linear phase'),
    'RAU': ('0.9FN minimum phase', '0.9FN linear phase')
}

# instrument and orientation codes matching sensor types
_instrument_orientation_code = {
    0: '',  # not defined
    1: 'DH',  # hydrophone
    2: 'HZ',  # geophone, vertical
    3: 'H1',  # geophone, horizontal, in-line
    4: 'H2',  # geophone, horizontal, crossline
    5: 'H3',  # geophone, horizontal, other
    6: 'NZ',  # accelerometer, vertical
    7: 'N1',  # accelerometer, horizontal, in-line
    8: 'N2',  # accelerometer, horizontal, crossline
    9: 'N3'   # accelerometer, horizontal, other
}


# band codes matching sample rate, for a short-period instrument
def _band_code(sample_rate):
    if sample_rate >= 1000:
        return 'G'
    if sample_rate >= 250:
        return 'D'
    if sample_rate >= 80:
        return 'E'
    if sample_rate >= 10:
        return 'S'


def _bcd(byte):
    """Decode 1-byte binary code decimals."""

    if isinstance(byte, (native_str, str)):
        try:
            byte = ord(byte)
        except TypeError:
            raise ValueError('not a byte')
    elif isinstance(byte, int):
        if byte > 255:
            raise ValueError('not a byte')
    else:
        raise ValueError('not a byte')
    v1 = byte >> 4
    v2 = byte & 0xF
    return v1, v2


def _decode_bcd(bytes_in):
    """Decode arbitrary length binary code decimals."""
    v = 0
    if isinstance(bytes_in, int):
        bytes_in = bytes([bytes_in])
    n = len(bytes_in)
    n = n*2 - 1  # 2 values per byte
    for byte in bytes_in:
        v1, v2 = _bcd(byte)
        v += v1*10**n + v2*10**(n-1)
        n -= 2
    return v


def _decode_bin(bytes_in):
    """Decode unsigned ints."""
    if isinstance(bytes_in, int):
        bytes_in = bytes([bytes_in])
    ll = len(bytes_in)
    # zero-pad to 4 bytes
    b = (chr(0)*(4-ll)).encode()
    b += bytes_in
    return unpack('>I', b)[0]


def _decode_bin_bool(bytes_in):
    """Decode unsigned ints as booleans."""
    b = _decode_bin(bytes_in)
    return b > 0


def _decode_fraction(bytes_in):
    """Decode positive binary fractions."""
    if PY2:
        # transform bytes_in to a list of ints
        bytes_ord = map(ord, bytes_in)
    else:
        # in PY3 this is already the case
        bytes_ord = bytes_in
    bit = ''.join('{:08b}'.format(b) for b in bytes_ord)
    return sum(int(x) * 2**-n for n, x in enumerate(bit, 1))


def _decode_flt(bytes_in):
    """Decode single-precision floats."""
    if isinstance(bytes_in, int):
        bytes_in = bytes([bytes_in])
    ll = len(bytes_in)
    # zero-pad to 4 bytes
    b = (chr(0)*(4-ll)).encode()
    b += bytes_in
    f = unpack('>f', b)[0]
    if math.isnan(f):
        f = None
    return f


def _decode_dbl(bytes_in):
    """Decode double-precision floats."""
    return unpack('>d', bytes_in)[0]


def _decode_asc(bytes_in):
    """Decode ascii."""
    if PY2:
        # transform bytes_in to a list of ints
        bytes_ord = map(ord, bytes_in)
    else:
        # in PY3 this is already the case
        bytes_ord = bytes_in
    printable = map(ord, string.printable)
    s = ''.join(chr(x) for x in bytes_ord if x in printable)
    if not s:
        s = None
    return s


def _read_ghb1(fp):
    """Read general header block #1."""
    buf = fp.read(32)
    ghb1 = OrderedDict()
    ghb1['file_number'] = _decode_bcd(buf[0:2])
    _format_code = _decode_bcd(buf[2:4])
    if _format_code != 8058:
        raise SEGDNotImplemented('Only 32 bit IEEE demultiplexed data '
                                 'is currently supported')
    ghb1['format_code'] = _decode_bcd(buf[2:4])
    ghb1['general_constants'] = [_decode_bcd(b) for b in buf[4:10]]  # unsure
    _year = _decode_bcd(buf[10:11]) + 2000
    _nblocks, _jday = _bcd(buf[11])
    ghb1['n_additional_blocks'] = _nblocks
    _jday *= 100
    _jday += _decode_bcd(buf[12:13])
    _hour = _decode_bcd(buf[13:14])
    _min = _decode_bcd(buf[14:15])
    _sec = _decode_bcd(buf[15:16])
    ghb1['time'] = UTCDateTime(year=_year, julday=_jday,
                               hour=_hour, minute=_min, second=_sec)
    ghb1['manufacture_code'] = _decode_bcd(buf[16:17])
    ghb1['manufacture_serial_number'] = _decode_bcd(buf[17:19])
    ghb1['bytes_per_scan'] = _decode_bcd(buf[19:22])
    _bsi = _decode_bcd(buf[22:23])
    if _bsi < 10:
        _bsi = 1./_bsi
    else:
        _bsi /= 10.
    ghb1['base_scan_interval_in_ms'] = _bsi
    _pol, _ = _bcd(buf[23])
    ghb1['polarity'] = _pol
    # 23L-24 : not used
    _rec_type, _rec_len = _bcd(buf[25])
    ghb1['record_type'] = _record_types[_rec_type]
    _rec_len = 0x100 * _rec_len
    _rec_len += _decode_bin(buf[26:27])
    if _rec_len == 0xFFF:
        _rec_len = None
    ghb1['record_length'] = _rec_len
    ghb1['scan_type_per_record'] = _decode_bcd(buf[27:28])
    ghb1['n_channel_sets_per_record'] = _decode_bcd(buf[28:29])
    ghb1['n_sample_skew_32bit_extensions'] = _decode_bcd(buf[29:30])
    ghb1['extended_header_length'] = _decode_bcd(buf[30:31])
    _ehl = _decode_bcd(buf[31:32])
    # If more than 99 External Header blocks are used,
    # then this field is set to FF and General Header block #2 (bytes 8-9)
    # indicates the number of External Header blocks.
    if _ehl == 0xFF:
        _ehl = None
    ghb1['external_header_length'] = _ehl
    return ghb1


def _read_ghb2(fp):
    """Read general header block #2."""
    buf = fp.read(32)
    ghb2 = OrderedDict()
    ghb2['expanded_file_number'] = _decode_bin(buf[0:3])
    # 3-6 : not used
    ghb2['external_header_blocks'] = _decode_bin(buf[7:9])
    # 9 : not used
    _rev = ord(buf[10:11])
    _rev += ord(buf[11:12])/10.
    ghb2['segd_revision_number'] = _rev
    ghb2['no_blocks_of_general_trailer'] = _decode_bin(buf[12:14])
    ghb2['extended_record_length_in_ms'] = _decode_bin(buf[14:17])
    # 17 : not used
    ghb2['general_header_block_number'] = _decode_bin(buf[18:19])
    # 19-32 : not used
    return ghb2


def _read_ghb3(fp):
    """Read general header block #3."""
    buf = fp.read(32)
    ghb3 = OrderedDict()
    ghb3['expanded_file_number'] = _decode_bin(buf[0:3])
    _sln = _decode_bin(buf[3:6])
    _sln += _decode_fraction(buf[6:8])
    ghb3['source_line_number'] = _sln
    _spn = _decode_bin(buf[8:11])
    _spn += _decode_fraction(buf[11:13])
    ghb3['source_point_number'] = _spn
    ghb3['phase_control'] = _decode_bin(buf[14:15])
    ghb3['vibrator_type'] = _decode_bin(buf[15:16])
    ghb3['phase_angle'] = _decode_bin(buf[16:18])
    ghb3['general_header_block_number'] = _decode_bin(buf[18:19])
    ghb3['source_set_number'] = _decode_bin(buf[19:20])
    # 20-32 : not used
    return ghb3


def _read_sch(fp):
    """Read scan type header."""
    buf = fp.read(32)
    # check if all the bytes are zero:
    if PY2:
        # convert buf to a list of ints
        _sum = sum(map(ord, buf))
    else:
        # in PY3 this is already the case
        _sum = sum(buf)
    if _sum == 0:
        raise SEGDScanTypeError('Empty scan type header')
    sch = OrderedDict()
    sch['scan_type_header'] = _decode_bcd(buf[0:1])
    sch['channel_set_number'] = _decode_bcd(buf[1:2])
    sch['channel_set_starting_time'] = _decode_bin(buf[2:4])
    sch['channel_set_end_time'] = _decode_bin(buf[4:6])
    _dm = _decode_bin(buf[6:8][::-1])
    sch['descale_multiplier_in_mV'] = _descale_multiplier[_dm]
    sch['number_of_channels'] = _decode_bcd(buf[8:10])
    _ctid, _ = _bcd(buf[10])
    sch['channel_type_id'] = _ctid
    _nse, _cgcm = _bcd(buf[11])
    sch['number_of_subscans_exponent'] = _nse
    sch['channel_gain_control_method'] = _cgcm
    sch['alias_filter_freq_at_-3dB_in_Hz'] = _decode_bcd(buf[12:14])
    sch['alias_filter_slope_in_dB/octave'] = _decode_bcd(buf[14:16])
    sch['low-cut_filter_freq_in_Hz'] = _decode_bcd(buf[16:18])
    sch['low-cut_filter_slope_in_dB/octave'] = _decode_bcd(buf[18:20])
    sch['first_notch_freq'] = _decode_bcd(buf[20:22])
    sch['second_notch_freq'] = _decode_bcd(buf[22:24])
    sch['third_notch_freq'] = _decode_bcd(buf[24:26])
    sch['extended_channel_set_number'] = _decode_bcd(buf[26:28])
    _ehf, _the = _bcd(buf[28])
    sch['extended_header_flag'] = _ehf
    sch['trace_header_extensions'] = _the
    sch['vertical_stack'] = _decode_bin(buf[29:30])
    sch['streamer_cable_number'] = _decode_bin(buf[30:31])
    sch['array_forming'] = _decode_bin(buf[31:32])
    return sch


def _read_extdh(fp, size):
    """Read extended header."""
    buf = fp.read(size)
    extdh = OrderedDict()
    # SERCEL extended header format
    extdh['acquisition_length_in_ms'] = _decode_bin(buf[0:4])
    extdh['sample_rate_in_us'] = _decode_bin(buf[4:8])
    extdh['total_number_of_traces'] = _decode_bin(buf[8:12])
    extdh['number_of_auxes'] = _decode_bin(buf[12:16])
    extdh['number_of_seis_traces'] = _decode_bin(buf[16:20])
    extdh['number_of_dead_seis_traces'] = _decode_bin(buf[20:24])
    extdh['number_of_live_seis_traces'] = _decode_bin(buf[24:28])
    _tos = _decode_bin(buf[28:32])
    extdh['type_of_source'] = _source_types[_tos]
    extdh['number_of_samples_in_trace'] = _decode_bin(buf[32:36])
    extdh['shot_number'] = _decode_bin(buf[36:40])
    extdh['TB_window_in_s'] = _decode_flt(buf[40:44])
    _trt = _decode_bin(buf[44:48])
    extdh['test_record_type'] = _test_record_types[_trt]
    extdh['spread_first_line'] = _decode_bin(buf[48:52])
    extdh['spread_first_number'] = _decode_bin(buf[52:56])
    extdh['spread_number'] = _decode_bin(buf[56:60])
    _st = _decode_bin(buf[60:64])
    extdh['spread_type'] = _spread_types[_st]
    extdh['time_break_in_us'] = _decode_bin(buf[64:68])
    extdh['uphole_time_in_us'] = _decode_bin(buf[68:72])
    extdh['blaster_id'] = _decode_bin(buf[72:76])
    extdh['blaster_status'] = _decode_bin(buf[76:80])
    extdh['refraction_delay_in_ms'] = _decode_bin(buf[80:84])
    extdh['TB_to_T0_time_in_us'] = _decode_bin(buf[84:88])
    extdh['internal_time_break'] = _decode_bin_bool(buf[88:92])
    extdh['prestack_within_field_units'] = _decode_bin_bool(buf[92:96])
    _net = _decode_bin(buf[96:100])
    extdh['noise_elimination_type'] = _noise_elimination_types[_net]
    extdh['low_trace_percentage'] = _decode_bin(buf[100:104])
    extdh['low_trace_value_in_dB'] = _decode_bin(buf[104:108])
    _value1 = _decode_bin(buf[108:112])
    _value2 = _decode_bin(buf[112:116])
    if _net == 2:
        # Diversity Stack
        extdh['number_of_windows'] = _value1
    elif _net == 3:
        # Historic
        extdh['historic_editing_type'] = _historic_editing_types[_value2]
        extdh['historic_range'] = _decode_bin(buf[120:124])
        extdh['historic_taper_length_2_exponent'] = _decode_bin(buf[124:128])
        extdh['historic_threshold_init_value'] = _decode_bin(buf[132:136])
        extdh['historic_zeroing_length'] = _decode_bin(buf[136:140])
    elif _net == 4:
        # Enhanced Diversity Stack
        extdh['window_length'] = _value1
        extdh['overlap'] = _value2
    extdh['noisy_trace_percentage'] = _decode_bin(buf[116:120])
    _thv = _decode_bin(buf[128:132])
    extdh['threshold_hold/var'] = _threshold_types[_thv]
    _top = _decode_bin(buf[140:144])
    extdh['type_of_process'] = _process_types[_top]
    extdh['acquisition_type_tables'] =\
        [_decode_bin(buf[144+n*4:144+(n+1)*4]) for n in range(32)]
    extdh['threshold_type_tables'] =\
        [_decode_bin(buf[272+n*4:272+(n+1)*4]) for n in range(32)]
    extdh['stacking_fold'] = _decode_bin(buf[400:404])
    # 404-483 : not used
    extdh['record_length_in_ms'] = _decode_bin(buf[484:488])
    extdh['autocorrelation_peak_time_in_ms'] = _decode_bin(buf[488:492])
    # 492-495 : not used
    extdh['correlation_pilot_number'] = _decode_bin(buf[496:500])
    extdh['pilot_length_in_ms'] = _decode_bin(buf[500:504])
    extdh['sweep_length_in_ms'] = _decode_bin(buf[504:508])
    extdh['acquisition_number'] = _decode_bin(buf[508:512])
    extdh['max_of_max_aux'] = _decode_flt(buf[512:516])
    extdh['max_of_max_seis'] = _decode_flt(buf[516:520])
    extdh['dump_stacking_fold'] = _decode_bin(buf[520:524])
    extdh['tape_label'] = _decode_asc(buf[524:540])
    extdh['tape_number'] = _decode_bin(buf[540:544])
    extdh['software_version'] = _decode_asc(buf[544:560])
    extdh['date'] = _decode_asc(buf[560:572])
    extdh['source_easting'] = _decode_dbl(buf[572:580])
    extdh['source_northing'] = _decode_dbl(buf[580:588])
    extdh['source_elevation'] = _decode_flt(buf[588:592])
    extdh['slip_sweep_mode_used'] = _decode_bin_bool(buf[592:596])
    extdh['files_per_tape'] = _decode_bin(buf[596:600])
    extdh['file_count'] = _decode_bin(buf[600:604])
    extdh['acquisition_error_description'] = _decode_asc(buf[604:764])
    _ft = _decode_bin(buf[764:768])
    extdh['filter_type'] = _filter_types[_ft]
    extdh['stack_is_dumped'] = _decode_bin_bool(buf[768:772])
    _ss = _decode_bin(buf[772:776])
    if _ss == 2:
        _ss = -1
    extdh['stack_sign'] = _ss
    extdh['PRM_tilt_correction_used'] = _decode_bin_bool(buf[776:780])
    extdh['swath_name'] = _decode_asc(buf[780:844])
    _om = _decode_bin(buf[844:848])
    # XXX: here I suppose that several operating modes are possible
    _op_mode = []
    for key in _operating_modes:
        try:
            k = _om & key
            _op_mode.append(_operating_modes[k])
        except KeyError:
            continue
    extdh['operating_mode'] = _op_mode
    # 848-851 : reserved
    extdh['no_log'] = _decode_bin_bool(buf[852:856])
    extdh['listening_time_in_ms'] = _decode_bin(buf[856:860])
    _tod = _decode_bin(buf[860:864])
    extdh['type_of_dump'] = _dump_types[_tod]
    # 864-867 : reserved
    extdh['swath_id'] = _decode_bin(buf[868:872])
    extdh['seismic_trace_offset_removal_is_disabled'] = \
        _decode_bin_bool(buf[872:876])
    _gps_microseconds = unpack('>Q', buf[876:884])[0]
    _gps_time = UTCDateTime('19800106') + _gps_microseconds/1e6
    # _gps_time includes leap seconds (17 as for November 2016)
    extdh['gps_time_of_acquisition'] = _gps_time
    # 884-963 : reserved
    # 964-1023 : not used
    return extdh


def _read_extrh(fp, size):
    """Read external header."""
    buf = fp.read(size)
    return _decode_asc(buf)


def _read_traceh(fp):
    """Read trace header."""
    buf = fp.read(20)
    traceh = OrderedDict()
    _fn = _decode_bcd(buf[0:2])
    if _fn == 0xFFFF:
        _fn = None
    traceh['file_number'] = _fn
    traceh['scan_type_number'] = _decode_bcd(buf[2:3])
    traceh['channel_set_number'] = _decode_bcd(buf[3:4])
    traceh['trace_number'] = _decode_bcd(buf[4:6])
    traceh['first_timing_word_in_ms'] = _decode_bin(buf[6:9]) * 1./256
    traceh['trace_header_extension'] = _decode_bin(buf[9:10])
    traceh['sample_skew'] = _decode_bin(buf[10:11])
    traceh['trace_edit'] = _decode_bin(buf[11:12])
    traceh['time_break_window'] = _decode_bin(buf[12:14])
    traceh['time_break_window'] += _decode_bin(buf[14:15])/100.
    traceh['extended_channel_set_number'] = _decode_bin(buf[15:16])
    traceh['extended_file_number'] = _decode_bin(buf[17:20])
    return traceh


def _read_traceh_eb1(fp):
    """Read trace header extension block #1, SEGD standard."""
    buf = fp.read(32)
    traceh = OrderedDict()
    _rln = _decode_bin(buf[0:3])
    if _rln == 0xFFFFFF:
        _rln = None
    traceh['receiver_line_number'] = _rln
    _rpn = _decode_bin(buf[3:6])
    if _rpn == 0xFFFFFF:
        _rpn = None
    traceh['receiver_point_number'] = _rpn
    traceh['receiver_point_index'] = _decode_bin(buf[6:7])
    traceh['number_of_samples_per_trace'] = _decode_bin(buf[7:10])
    _erln = _decode_bin(buf[10:13])
    _frac = _decode_fraction(buf[13:15])
    traceh['extended_receiver_line_number'] = _erln + _frac
    _erpn = _decode_bin(buf[15:18])
    _frac = _decode_fraction(buf[18:20])
    traceh['extended_receiver_point_number'] = _erpn + _frac
    _sensor_code = _decode_bin(buf[20:21])
    traceh['sensor_code'] = _sensor_code  # not in spec: for internal use
    traceh['sensor_type'] = _sensor_types[_sensor_code]
    # 21-31 : not used
    return traceh


def _read_traceh_eb2(fp):
    """Read trace header extension block #2, SERCEL format."""
    buf = fp.read(32)
    traceh = OrderedDict()
    traceh['receiver_point_easting'] = _decode_dbl(buf[0:8])
    traceh['receiver_point_northing'] = _decode_dbl(buf[8:16])
    traceh['receiver_point_elevation'] = _decode_flt(buf[16:20])
    traceh['sensor_type_number'] = _decode_bin(buf[20:21])
    # 21-23 : not used
    traceh['DSD_identification_number'] = _decode_bin(buf[24:28])
    traceh['extended_trace_number'] = _decode_bin(buf[28:32])
    return traceh


def _read_traceh_eb3(fp):
    """Read trace header extension block #3, SERCEL format."""
    buf = fp.read(32)
    traceh = OrderedDict()
    traceh['resistance_low_limit'] = _decode_flt(buf[0:4])
    traceh['resistance_high_limit'] = _decode_flt(buf[4:8])
    traceh['resistance_calue_in_ohms'] = _decode_flt(buf[8:12])
    traceh['tilt_limit'] = _decode_flt(buf[12:16])
    traceh['tilt_value'] = _decode_flt(buf[16:20])
    traceh['resistance_error'] = _decode_bin_bool(buf[20:21])
    traceh['tilt_error'] = _decode_bin_bool(buf[21:22])
    # 22-31 : not used
    return traceh


def _read_traceh_eb4(fp):
    """Read trace header extension block #4, SERCEL format."""
    buf = fp.read(32)
    traceh = OrderedDict()
    traceh['capacitance_low_limit'] = _decode_flt(buf[0:4])
    traceh['capacitance_high_limit'] = _decode_flt(buf[4:8])
    traceh['capacitance_value_in_nano_farads'] = _decode_flt(buf[8:12])
    traceh['cutoff_low_limit'] = _decode_flt(buf[12:16])
    traceh['cutoff_high_limit'] = _decode_flt(buf[16:20])
    traceh['cutoff_value_in_Hz'] = _decode_flt(buf[20:24])
    traceh['capacitance_error'] = _decode_bin_bool(buf[24:25])
    traceh['cutoff_error'] = _decode_bin_bool(buf[25:26])
    # 26-31 : not used
    return traceh


def _read_traceh_eb5(fp):
    """Read trace header extension block #5, SERCEL format."""
    buf = fp.read(32)
    traceh = OrderedDict()
    traceh['leakage_limit'] = _decode_flt(buf[0:4])
    traceh['leakage_value_in_megahoms'] = _decode_flt(buf[4:8])
    traceh['instrument_longitude'] = _decode_dbl(buf[8:16])
    traceh['instrument_latitude'] = _decode_dbl(buf[16:24])
    traceh['leakage_error'] = _decode_bin_bool(buf[24:25])
    traceh['instrument_horizontal_position_accuracy_in_mm'] = \
        _decode_bin(buf[25:28])
    traceh['instrument_elevation_in_mm'] = _decode_flt(buf[28:32])
    return traceh


def _read_traceh_eb6(fp):
    """Read trace header extension block #6, SERCEL format."""
    buf = fp.read(32)
    traceh = OrderedDict()
    _ut = _decode_bin(buf[0:1])
    traceh['unit_type'] = _unit_types[_ut]
    traceh['unit_serial_number'] = _decode_bin(buf[1:4])
    traceh['channel_number'] = _decode_bin(buf[4:5])
    # 5-7 : not used
    traceh['assembly_type'] = _decode_bin(buf[8:9])
    traceh['assembly_serial_number'] = _decode_bin(buf[9:12])
    traceh['location_in_assembly'] = _decode_bin(buf[12:13])
    # 13-15 : not used
    _st = _decode_bin(buf[16:17])
    traceh['subunit_type'] = _subunit_types[_st]
    _ct = _decode_bin(buf[17:18])
    traceh['channel_type'] = _channel_types[_ct]
    # 18-19 : not used
    traceh['sensor_sensitivity_in_mV/m/s/s'] = _decode_flt(buf[20:24])
    # 24-31 : not used
    return traceh


def _read_traceh_eb7(fp):
    """Read trace header extension block #7, SERCEL format."""
    buf = fp.read(32)
    traceh = OrderedDict()
    _cut = _decode_bin(buf[0:1])
    traceh['control_unit_type'] = _control_unit_types[_cut]
    traceh['control_unit_serial_number'] = _decode_bin(buf[1:4])
    traceh['channel_gain_scale'] = _decode_bin(buf[4:5])
    traceh['channel_filter'] = _decode_bin(buf[5:6])
    traceh['channel_data_error_overscaling'] = _decode_bin(buf[6:7])
    _ces = _decode_bin(buf[7:8])
    traceh['channel_edited_status'] = _channel_edited_statuses[_ces]
    traceh['channel_sample_to_mV_conversion_factor'] = _decode_flt(buf[8:12])
    traceh['number_of_stacks_noisy'] = _decode_bin(buf[12:13])
    traceh['number_of_stacks_low'] = _decode_bin(buf[13:14])
    _channel_type_ids = {1: 'seis', 9: 'aux'}
    _cti = _decode_bin(buf[14:15])
    traceh['channel_type_id'] = _channel_type_ids[_cti]
    _cp = _decode_bin(buf[15:16])
    traceh['channel_process'] = _channel_processes[_cp]
    traceh['trace_max_value'] = _decode_flt(buf[16:20])
    traceh['trace_max_time_in_us'] = _decode_bin(buf[20:24])
    traceh['number_of_interpolations'] = _decode_bin(buf[24:28])
    traceh['seismic_trace_offset_value'] = _decode_bin(buf[28:32])
    return traceh


def _read_trace_data(fp, size):
    buf = array('f')
    # buf.fromfile(fp, size) doesn't work with py2
    buf.fromstring(fp.read(size*4))
    buf.byteswap()
    buf = np.array(buf, dtype=np.float32)
    return buf


def _read_trace_data_block(fp, size):
    traceh = _read_traceh(fp)
    th_ext = traceh['trace_header_extension']
    _read_traceh_eb = [_read_traceh_eb1, _read_traceh_eb2,
                       _read_traceh_eb3, _read_traceh_eb4,
                       _read_traceh_eb5, _read_traceh_eb6,
                       _read_traceh_eb7]
    for n in range(th_ext):
        traceh.update(_read_traceh_eb[n](fp))
    _cgs = traceh['channel_gain_scale'] - 1
    _st = traceh['subunit_type']
    # sanity check against corrupted values for 'subunit_type'
    if _st in _channel_gain_scales.keys():
        _unit = _channel_gain_units[_st]
        traceh['channel_gain_scale_in_' + _unit] =\
            _channel_gain_scales[_st][_cgs]
        del traceh['channel_gain_scale']
        _cf = traceh['channel_filter'] - 1
        traceh['channel_filter'] = _channel_filters[_st][_cf]

    data = _read_trace_data(fp, size)
    return traceh, data


def _build_segd_header(generalh, sch, extdh, extrh, traceh):
    segd = OrderedDict()
    segd.update(generalh)
    channel_set_number = traceh['channel_set_number']
    segd.update(sch[channel_set_number])
    segd.update(extdh)
    segd['external_header'] = extrh
    segd.update(traceh)
    # remove fileds only useful for internal use
    del segd['file_number']
    del segd['extended_file_number']
    del segd['expanded_file_number']
    del segd['format_code']
    del segd['bytes_per_scan']
    del segd['extended_header_length']
    del segd['external_header_length']
    del segd['external_header_blocks']
    del segd['scan_type_number']
    del segd['channel_set_number']
    del segd['number_of_subscans_exponent']
    del segd['extended_channel_set_number']
    del segd['channel_number']
    del segd['trace_number']
    del segd['extended_trace_number']
    del segd['extended_header_flag']
    del segd['number_of_channels']
    del segd['number_of_seis_traces']
    del segd['number_of_live_seis_traces']
    del segd['number_of_dead_seis_traces']
    del segd['number_of_auxes']
    del segd['total_number_of_traces']
    del segd['sensor_code']
    del segd['sensor_type_number']
    del segd['trace_header_extension']
    del segd['trace_header_extensions']
    del segd['max_of_max_seis']
    del segd['max_of_max_aux']
    # remove fields used in trace.stats
    # or that can be evaluated from trace data
    del segd['time']
    del segd['sample_rate_in_us']
    del segd['number_of_samples_in_trace']
    del segd['number_of_samples_per_trace']
    del segd['record_length_in_ms']
    del segd['extended_record_length_in_ms']
    del segd['acquisition_length_in_ms']
    del segd['trace_max_time_in_us']
    del segd['trace_max_value']
    # remove None values
    segd = {k: v for k, v in segd.items() if v is not None}
    return segd


# def _print_dict(dict, title):
#     print(title)
#     for key, val in dict.iteritems():
#         print('{}: {}'.format(key, val))


def read_segd(filename):
    fp = open(filename, 'rb')
    generalh = _read_ghb1(fp)
    generalh.update(_read_ghb2(fp))
    generalh.update(_read_ghb3(fp))
    sch = {}
    for n in range(generalh['n_channel_sets_per_record']):
        try:
            _sch = _read_sch(fp)
        except SEGDScanTypeError:
            continue
        sch[_sch['channel_set_number']] = _sch
    size = generalh['extended_header_length']*32
    extdh = _read_extdh(fp, size)
    ext_hdr_lng = generalh['external_header_length']
    if ext_hdr_lng == 0xFF:
        ext_hdr_lng = generalh['external_header_blocks']
    size = ext_hdr_lng*32
    extrh = _read_extrh(fp, size)
    sample_rate = extdh['sample_rate_in_us']/1e6
    npts = extdh['number_of_samples_in_trace']
    size = npts
    st = Stream()
    convert_to_int = True
    for n in range(extdh['total_number_of_traces']):
        traceh, data = _read_trace_data_block(fp, size)
        # check if all traces can be converted to int
        convert_to_int = convert_to_int and np.all(np.mod(data, 1) == 0)
        # _print_dict(traceh, '***TRACEH:')
        tr = Trace(data)
        tr.stats.station = str(traceh['unit_serial_number'])
        tr.stats.channel = _band_code(1./sample_rate)
        tr.stats.channel += _instrument_orientation_code[traceh['sensor_code']]
        tr.stats.delta = sample_rate
        tr.stats.starttime = generalh['time']
        tr.stats.segd = _build_segd_header(generalh, sch, extdh, extrh, traceh)
        st.append(tr)
    fp.close()
    # for n, _sch in sch.iteritems():
    #     _print_dict(_sch, '***SCH %d:' % n)
    # _print_dict(extdh, '***EXTDH:')
    # print('***EXTRH:\n %s' % extrh)
    # _print_dict(generalh, '***GENERALH:')
    if convert_to_int:
        for tr in st:
            tr.data = tr.data.astype(np.int32)
    return st


if __name__ == '__main__':
    import sys
    st = read_segd(sys.argv[1])
    print(st)
    st.write('test.mseed', format='MSEED')
    # for tr in st:
    # tr = st[0]
    # _print_dict(tr.stats.segd, '')
    #     print tr.stats
    # st.plot()
