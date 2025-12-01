#!/usr/bin/env python3
# encoding: utf-8

r"""
yaml2ck.py: Convert Cantera YAML input files to Chemkin-format mechanisms, thermo data and transport data.

Usage:
------
python yaml2ck [--input=<filename>] 
               [--output=<filename>]
               [--thermo=<filename>]
               [--transport=<filename>]

Examples
--------
>>> python yaml2ck --input='gri.yaml' --output='gri.inp'

>>> python yaml2ck --input='gri.yaml'

>>> python yaml2ck -h

The input mechanism input file format could be `.yaml` , `.cti` or `.xml` .

Note: The required cantera version is above 2.6.0 . 

"""
## system-level import
import sys, os
import cantera as ct
import logging
import getopt
# from argparse import ArgumentParser

## user-level import
from .mechanism_convert import *


logger = logging.getLogger(__name__)
loghandler = logging.StreamHandler(sys.stdout)
logformatter = logging.Formatter('%(message)s')
loghandler.setFormatter(logformatter)
logger.handlers.clear()
logger.addHandler(loghandler)
logger.setLevel(logging.INFO)



def main(argv):

    longOptions = ["input=", "output=", "thermo=", "transport=", "help"]

    try:
        optlist, args = getopt.getopt(argv, 'dh', longOptions)
        options = dict()
        for o,a in optlist:
            options[o] = a

        if args:
            raise getopt.GetoptError('Unexpected command line option: ' +
                                     repr(' '.join(args)))

    except getopt.GetoptError as e:
        logger.error('yaml2ck.py: Error parsing arguments:')
        logger.error(e)
        logger.error('Run "python yaml2ck --help" or "python yaml2ck -h" to see usage help.')
        sys.exit(1)
    if not options or '-h' in options or '--help' in options:
        logger.info(__doc__)
        sys.exit(0)


    mech_path = options.get("--input")
    
    basename = os.path.splitext(mech_path)[0]

    if '--output' in options:
        output = options['--output']
    else:
        output = basename + "_mech.inp"
    
    if '--thermo' in options:
        thermo = options['--thermo']
    else:
        thermo = basename + '_thermo.dat'

    if '--transport' in options:
        transport = options['--transport']
    else:
        transport = basename + '_transport.dat'

    if not os.path.exists(mech_path):
        logger.error(f"Error: the mechnanism input file \033[31m{mech_path}\033[0m is not found.")
        sys.exit(0)
    
    gas = ct.Solution(mech_path)
    spec = ct.Species.list_from_file(mech_path)
    write_chem_data(gas, output_filename=output)
    write_thermo_data(spec, thermo)
    write_transport_data(spec, transport)
    logger.info(f"input mechanism file path: \033[36m{mech_path}\033[0m")
    logger.info(f"output CHEMKIN file path:  \033[32m{output}\033[0m")
    logger.info(f"thermo data path:          \033[32m{thermo}\033[0m")
    logger.info(f"transport data path:       \033[32m{transport}\033[0m")
    logger.info("\033[1;32mCONVERTION FINISHED\033[0m")

def script_entry_point():
    main(sys.argv[1:])

if __name__ == '__main__':
    main(sys.argv[1:])