#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Antonio Valentino, Forrest Williams, Aug 2022    #
############################################################


import sys

from mintpy.utils.arg_utils import create_argument_parser

#########################################################################
NOTE = """
  For each interferogram, the unwrapped interferogram, coherence, and metadata the file name is required e.g.:
  1) date1-date2.geo.unw.tif
  2) date1-date2.cc.unw.tif
  3) metadata.txt

  A DEM filename is needed and a E,N,U filename is recommended  e.g.:
  1) track_ID.geo.hgt.tif 
  2) track_ID.geo.E.tif 
  3) track_ID.geo.N.tif 
  4) track_ID.geo.U.tif

  This script will read these files, read the geospatial metadata from GDAL,
  use general metadata file (for interferograms and coherence),
  and write to a ROI_PAC .rsc metadata file with the same name as the input file with suffix .rsc,
  e.g. date1-date2.geo.unw.tiff.rsc

  Here is an example of how your HyP3 files should look:

  Before loading:
      For each interferogram, 2 files are needed:
          date1-date2.geo.unw.tif
          date1-date2.geo.cc.tif
      For the geometry file 4 file are recommended:
          track_ID.geo.hgt.tif 
          track_ID.geo.E.tif 
          track_ID.geo.N.tif 
          track_ID.geo.U.tif

  After running prep_licsar.py:
      For each interferogram:
          date1-date2.geo.unw.tif
          date1-date2.geo.unw.tif.rsc
          date1-date2.geo.cc.tif
          date1-date2.geo.cc.tif.rsc
      For the input geometry files:
          track_ID.geo.hgt.tif
          track_ID.geo.hgt.tif.rsc 
          track_ID.geo.E.tif
          track_ID.geo.E.tif.rsc
          track_ID.geo.N.tif 
          track_ID.geo.N.tif.rsc
          track_ID.geo.U.tif
          track_ID.geo.U.tif.rsc

  Notes:
        LICSAR new version of data has new header data which can be integrated but here is not implemented , the data suffix is .metadata
"""

EXAMPLE = """example:
  prep_licsar.py  /GEOC/*/*geo.unw.tif
  prep_licsar.py  /GEOC/*/*geo.cc.tif
  prep_licsar.py  *geo.E.tif
  prep_licsar.py  *geo.U.tif
  prep_licsar.py  *geo.N.tif
  prep_licsar.py  *geo.N.tif
  prep_licsar.py  *geo.hgt.tif
"""


def create_parser(subparsers=None):
    synopsis = 'Prepare attributes file for LiCSAR InSAR product.'
    epilog = EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis+NOTE, epilog=epilog, subparsers=subparsers)

    parser.add_argument('file', nargs='+', help='LiCSAR file(s)')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


#########################################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from mintpy.prep_licsar import prep_licsar

    # run
    prep_licsar(inps)


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
