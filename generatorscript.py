import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xml.etree.ElementTree as ET # XML PARSER. Commands can be found at https://docs.python.org/3/library/xml.etree.elementtree.html
from inputfilemanager import InputGenerator


# Parameters for best estimate
tend=100.
KerogenElementaryComp=np.array([159.,180.,16.,3.,2.])
# DecompStoichiometry=np.array([[1.,5.,5.,1.],[1.,1.,3.04348,0.04348]])
DecompStoichiometry=np.array([[1.,5.,5.,1.],[0.,23.,70.,1.]])
DecompFrequency=np.array([5.0e16,5.0e15]) # frequency s^-1
DecompActivationEnergy=np.array([245.,245.*1.1])*1.0e3 # Activation energy kJ/mol * 1000J/kJ
DecompFreeVolume=33.0e-6 # m^3
KerogenDensity=1.46 # g/cc
CokeDensity=2.97 # g/cc
tcooldown=7.5 # years
HeatDecaytsteps=20
KerVFracInit=0.1
PTIsothermtreaction=5. #yrs
porosity=0.3
reservoirthickness=120. # m
reservoirdepth=120. # m
wastedepth=40. # m
wastethickness=40. # m
wasteradius=0.5 # m
makeplots=False
nx=100
nz=120


Bestest=InputGenerator(tend, KerogenElementaryComp,KerogenDensity,CokeDensity,\
    DecompStoichiometry,DecompFrequency,DecompActivationEnergy, DecompFreeVolume, \
    tcooldown,HeatDecaytsteps,KerVFracInit,PTIsothermtreaction,porosity,\
    reservoirdepth,reservoirthickness,wastedepth,wastethickness,nx,nz)
Bestest.Compute_HeatRadiation(makeplot=makeplots)
Bestest.Compute_HeatOfReaction()
Bestest.Compute_PTIsothermLinearEqn(0.1,makeplots=makeplots)
Bestest.Import_BaseXML()
Bestest.Update_XML('BestEstimate.xml')



# # Parameters for conservative estimate
# tend=100.
# KerogenElementaryComp=np.array([159.,180.,16.,3.,2.])
# DecompStoichiometry=np.array([[1.,3.,2.,1.],\
#                             [0.,11.,33.,1.]])
# DecompFrequency=np.array([5.0e16,5.0e15]) # frequency s^-1
# DecompActivationEnergy=np.array([245.,245.*1.1])*1.0e3 # Activation energy kJ/mol * 1000J/kJ
# DecompFreeVolume=33.0e-6 # m^3
# tcooldown=1.0 # years
# HeatDecaytsteps=20
# KerVFracInit=0.1
# PTIsothermtreaction=5. #yrs
# porosity=0.3
# reservoirthickness=60. # m
# wastedepth=48. # m
# wastethickness=5. # m
# wasteradius=0.5 # m
# nx=100
# nz=20
