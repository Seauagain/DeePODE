
r"""
ct2foam.py: Convert Cantera zero-dimensional reaction or one-dimensional premixed laminar flame fields to openFOAM-format fields.

Usage:
------
python ct2foam [--mech=<mechanism path>] 
               [--fuel=<fuelname>]
               [--case_name=<case home path, default "EBIcase_1d">]
               [--timefold=<timestep path of initial field files, default "0">]
               [--T=<initial temperature>]
               [--P=<initial pressure (atm)>]
               [--Phi=<initial equivalence ratio>]
               [--keep_csv]
               [--0d]
               [--1d]

Examples
--------
>>> python ct2foam --mech=Chem/DME39_ct26.yaml --fuel=CH3OCH3 --T=1000 --P=1 --Phi=1 --0d 

>>> python ct2foam --mech=Chem/DME39_ct26.yaml --fuel=CH3OCH3 --T=600 --P=1 --Phi=1 --1d 

>>> python ct2foam --mech=Chem/DME39_ct26.yaml --fuel=CH3OCH3 --case_name=DME_1d --timefold=0 --T=600 --P=1 --Phi=1 --1d 

>>> python ct2foam --mech=Chem/DME39_ct26.yaml --fuel=CH3OCH3 --timefold=0 --T=600 --P=1 --Phi=1 --1d 

>>> python ct2foam -h
"""

## system-level import
import sys, os
import cantera as ct
import logging
import getopt

## user-level import
from .adiabatic_flame import *


logger = logging.getLogger(__name__)
loghandler = logging.StreamHandler(sys.stdout)
logformatter = logging.Formatter('%(message)s')
loghandler.setFormatter(logformatter)
logger.handlers.clear()
logger.addHandler(loghandler)
logger.setLevel(logging.INFO)


def main(argv):

    longOptions = ["mech=", "fuel=", "case_name=", "timefold=", "T=", "P=", "Phi=", "fuel=", "0d", "1d", "keep_csv", "help"]

    try:
        optlist, args = getopt.getopt(argv, 'dh', longOptions)
        options = dict()
        for o,a in optlist:
            options[o] = a

        if args:
            raise getopt.GetoptError('Unexpected command line option: ' +
                                        repr(' '.join(args)))
    
    except getopt.GetoptError as e:
        logger.error('ct2foam.py: Error parsing arguments:')
        logger.error(e)
        logger.error('Run "python ct2foam --help" or "python ct2foam -h" to see usage help.')
        sys.exit(1)
    if not options or '-h' in options or '--help' in options:
        logger.info(__doc__)
        sys.exit(0)
    
    if ("--0d" not in options) and ("--1d" not in options):
        logger.info(__doc__)
        sys.exit(0)
    
    mech_path = options.get("--mech")
    fuel = options.get("--fuel")
    if '--casename' in options:
        case_name = options.get("--case_name")
    else:
        case_name = ""
    
    if '--timefold' in options:
        timefold = options.get("--timefold")
    else:
        timefold = "0"
    T = float(options.get("--T"))
    P = float(options.get("--P"))
    Phi = float(options.get("--Phi"))

    
    if "--0d" in options:
        convert2FoamZeroDimFields(case_name, timefold, mech_path, fuel, T, P, Phi)


    if "--1d" in options:
        createOneDimFields(mech_path, fuel, T, P, Phi)
        convert2FoamOneDimFileds(case_name, timefold, mech_path, fuel, T, P, Phi)
        if "--keep_csv" not in options:
            cleanCSV(fuel, T, P, Phi)
    


def script_entry_point():
    main(sys.argv[1:])

if __name__ == '__main__':
    main(sys.argv[1:])