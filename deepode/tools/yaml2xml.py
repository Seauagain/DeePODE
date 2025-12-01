#!/usr/bin/env python3
# encoding: utf-8

r"""
yaml2xml.py: Convert Cantera YAML input files to XML-format mechanisms.

Usage:
------
python yaml2xml [--input=<filename>] 
                [--output=<filename>]

Examples
--------
>>> python yaml2xml --input='gri.yaml' --output='gri_custom.xml'
XML file saved in gri_custom.xml

>>> python yaml2xml --input='gri.yaml'
XML file saved in gri.xml

>>> python yaml2xml -h

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

    longOptions = ["input=", "output=", "help"]

    try:
        optlist, args = getopt.getopt(argv, 'dh', longOptions)
        options = dict()
        for o,a in optlist:
            options[o] = a

        if args:
            raise getopt.GetoptError('Unexpected command line option: ' +
                                     repr(' '.join(args)))

    except getopt.GetoptError as e:
        logger.error('yaml2xml.py: Error parsing arguments:')
        logger.error(e)
        logger.error('Run "python yaml2xml --help" or "python yaml2xml -h" to see usage help.')
        sys.exit(1)
    if not options or '-h' in options or '--help' in options:
        logger.info(__doc__)
        sys.exit(0)


    mech_path = options.get("--input")
    
    basename = os.path.splitext(mech_path)[0]

    inp_output = basename + "_mech.inp"

    if '--output' in options:
        output = options['--output']
    else:
        output = basename + ".xml"

    thermo = basename + '_thermo.dat'
    transport = basename + '_transport.dat'

    if not os.path.exists(mech_path):
        logger.error(f"Error: the mechnanism input file \033[31m{mech_path}\033[0m is not found.")
        sys.exit(0)
    
    gas = ct.Solution(mech_path)
    spec = ct.Species.list_from_file(mech_path)
    write_chem_data(gas, output_filename=inp_output)
    write_thermo_data(spec, thermo)
    write_transport_data(spec, transport)

    xml_name = inp_output.split(".")[0]
    os.system(f"ck2cti --input {inp_output} --thermo {thermo} --transport {transport} --permissive")
    os.system(f"ctml_writer {xml_name}.cti {output}")

    
    logger.info(f"\nRemove temporaty file:  \033[32m{xml_name}.cti\033[0m")
    logger.info(f"Remove temporaty file:  \033[32m{inp_output}\033[0m")
    logger.info(f"Remove temporaty file:  \033[32m{thermo}\033[0m")
    logger.info(f"Remove temporaty file:  \033[32m{transport}\033[0m")
    os.remove(inp_output)
    os.remove(f"{xml_name}.cti")
    os.remove(thermo)
    os.remove(transport)
    logger.info("\033[1;32mCONVERTION FINISHED\033[0m")
    logger.info(f"XML file saved in       \033[32m{output}\033[0m")
    

def script_entry_point():
    main(sys.argv[1:])

if __name__ == '__main__':
    main(sys.argv[1:])