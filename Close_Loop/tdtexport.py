import sys
sys.path += ['.']

import argparse

import tdt

from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description='Export TDT stream data to interlaced binary file, one file per store', formatter_class=RawTextHelpFormatter)

parser.add_argument('blockpath', type=str, nargs=1,
                    help='''data path to export''')
parser.add_argument('--format', dest='export', type=str, nargs=1, default=['interlaced'], required=False,
                    choices = ['csv','binary','interlaced'],
                    help='''exported data format. (default: interlaced)
csv:\t\tdata export to comma-separated value files
                streams: one file per store, one channel per column
                epocs: one column onsets, one column offsets
binary:\tstreaming data is exported as raw binary files
                one file per channel per store
interlaced:\tstreaming data exported as raw binary files
                one file per store, data is interlaced''')
parser.add_argument('--scale', type=float, nargs=1, default=1.0, required=False,
                    help='''scale data before export (default: %(default)s)''')
parser.add_argument('--dtype', type=str, nargs=1, default=None, required=False,
                    choices = ['i16','f32'],
                    help='''override data format to export, i16 or f32.
otherwise, the data format for each store in tank is used''')
parser.add_argument('--stores', type=str, nargs='+', default='', required=False,
                    help='''string or list, specify a single store or array of stores to extract''')
parser.add_argument('--channels', type=int, nargs='+', default=0, required=False,
                    help='''integer or list, choose a single channel or array of channels
to extract from stream or snippet events''')
parser.add_argument('--outdir', type=str, nargs=1, default=None, required=False,
                    help='''output directory for exported files. Defaults to current block folder
if not specified''')
parser.add_argument('--verbose', type=bool, nargs=1, default=False, required=False,
                    choices = ['True','False'],
                    help='''print extra debugging statements (default is False)''')

args = parser.parse_args()
verbose = args.verbose
print()
if verbose:
    print(args)
    print()

blockpath = args.blockpath[0]
export = args.export[0]
try:
    len(args.scale)
    scale = args.scale[0]
except:
    scale = args.scale

if args.dtype:
    dtype = args.dtype[0]
else:
    dtype = None
store = args.stores
channel = args.channels
if args.outdir:
    outdir = args.outdir[0]
    outdir_txt = outdir
else:
    outdir = None
    outdir_txt = blockpath

store_txt = store
if store_txt == '':
    store_txt = 'all'
    
channel_txt = repr(channel)
if channel_txt == '0':
    channel_txt = 'all'
if dtype is None:
    dtype_txt = 'default'
else:
    dtype_txt = dtype

verbose = args.verbose

ccc = 15
print('Exporting:'.rjust(ccc), blockpath)
print('To:'.rjust(ccc), outdir_txt)
print('Format:'.rjust(ccc), export)
print('Version:'.rjust(ccc), tdt.__version__)
print('Parameters:'.rjust(ccc))

ddd = ccc+12
print('scale:'.rjust(ddd), scale)
print('dtype:'.rjust(ddd), dtype_txt)
print('stores:'.rjust(ddd), store_txt)
print('channels:'.rjust(ddd), channel_txt)
print()

if verbose:
    print('tdt.read_block inputs:')
    print('blockpath:'.rjust(ccc),blockpath)
    print('export:'.rjust(ccc),export)
    print('channel:'.rjust(ccc),channel)
    print('store:'.rjust(ccc),store)
    print('outdir:'.rjust(ccc),outdir)
    print('scale:'.rjust(ccc),scale)
    print('dtype:'.rjust(ccc),dtype)
    print('verbose:'.rjust(ccc),verbose)
    print()

tdt.read_block(blockpath, export=export, channel=channel, store=store, outdir=outdir, scale=scale, dtype=dtype, verbose=verbose)
