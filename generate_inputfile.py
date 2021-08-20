import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xml.etree.ElementTree as ET # XML PARSER. Commands can be found at https://docs.python.org/3/library/xml.etree.elementtree.html

mpl.rcParams['axes.linewidth'] = 2.
mpl.rcParams['axes.edgecolor'] = 'gray'
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.family'] = 'Arial'

class InputGenerator(object):
    """docstring for InputGenerator.
    This object manages the data used to automatically generate input files. It
    takes as input the following:

    tend                    = End of simulation. Units : years (is converted to
                            seconds when generating the input file).

    KerogenElementaryComp   = Number of atoms for each element in the kerogen
                            model molecule. Should be in the order:
                            Carbon Hydrogen Oxygen Nitrogen Sulfur

    KerogenDensity          = Kerogen density. Units: g/cc or kg/L or tonne/m^3

    CokeDensity             = Coke (residual solid after pyrolysis) density.
                            Units: g/cc or kg/L or tonne/m^3

    DecompStoichiometry     = Stoichiometric coefficients in the reactions of
                            kerogen to oil in the first row, then oil to gas in
                            the second row. The order of stoichiometric coefficients
                            is the same as the order for KerogeenElementaryComp.

    DecompFrequency         = Frequency parameter A used to compute the first-order
                            reaction constant k=-Aexp(-(E+PV)/RT) in the reactions of
                            kerogen to oil and oil to gas. Units: s^-1 or Hertz

    DecompActivationEnergy  = Activation energy parameter E used to compute the first-order
                            reaction constant k=-Aexp(-(E+PV)/RT) in the reactions of
                            kerogen to oil and oil to gas. Units: J/mol

    DecompFreeVolume        = Volume parameter V used to compute the first-order
                            reaction constant k=-Aexp(-(E+PV)/RT) in the reactions of
                            kerogen to oil and oil to gas. Units: m^3

    tcooldown               = Cool down time between reaction and disposal.
                            Ewing (2015) says this should be 5-10 years.
                            Units: years
                            Ewing, R.C., 2015. Long-term storage of spent nuclear
                            fuel. Nature Materials, 14(3), pp.252-257.

    HeatDecaytsteps         = Number of rows in the tabulated decaying heat
                            radiation.

    PorosityInit            = Initial porosity. Does not include the volume
                            fraction occupied by the kerogen. Units: fraction

    KerVFracInit            = Initial rock volume fraction occupied by the kerogen.
                            Units: fraction.

    PTIsothermtreaction     = Time of reaction used in the kinetic model to
                            compute the PT isotherm line at a given mass fraction.
                            Units: years

    reservoirthickness      = Total thickness of the reservoir. Units: m

    reservoirdepth          = Total depth of the reservoir. Units: m

    wastedepth              = Depth of the reservoir, measured as the distance
                            between the top of the reservoir and the top of the
                            waste cannisters . Units: m

    wastethickness          = Total thickness of the heat-emitting waste cannisters.
                            Units: m

    wasteradius             = Total radius of the waste cannisters. Must be less
                            than the wellbore radius. Units: m

    nx                      = Number of grid blocks in the x-direction.

    nz                      = Number of grid blocks in the z-direction.
    """

    def __init__(self, tend, KerogenElementaryComp,KerogenDensity,CokeDensity, \
        DecompStoichiometry,DecompFrequency,DecompActivationEnergy, DecompFreeVolume, \
        tcooldown, HeatDecaytsteps,KerVFracInit,PTIsothermtreaction,PorosityInit,\
        reservoirdepth,reservoirthickness,wastedepth,wastethickness,nx,nz,\
        wasteradius=0.5,FuelSource='Uranium'):
        self.tend = tend
        self.zker = KerogenElementaryComp
        self.rhoker = KerogenDensity
        self.rhocoke = CokeDensity
        self.s = DecompStoichiometry
        self.A = DecompFrequency
        self.E = DecompActivationEnergy
        self.V = DecompFreeVolume
        self.tcd = tcooldown
        self.tsteps = HeatDecaytsteps
        self.tr = PTIsothermtreaction
        self.KerVFrac = KerVFracInit
        self.por = PorosityInit
        self.resh=reservoirthickness
        self.resd=reservoirdepth
        self.wah=wastethickness
        self.war=wasteradius
        self.wad=wastedepth
        self.nx=nx
        self.ny=1
        self.nz=nz
        self.zoil=np.array([10.,20.,0.,0.,0.])
        self.zgas=np.array([2.,6.,0.,0.,0.])
        self.R=8.3144598 # Ideal Gas constant
        self.Rada=-0.7467
        self.Radc=20.23e3
        self.Rad1Month=98489.# Hedin reports that after 40 years the heat output
        # rate was 1300 W per ton of Uranium. This corresponds to ~1.3% of the heat
        # generation rate measured 1 month after the nuclear waste exits the reactor.
        # https://www.osti.gov/etdeweb/servlets/purl/587853
        # The model for the heat decay in the following months is also fit to
        # data provided by Hedin.

        # The bottom should follow the order: Kerogen, Oil, Gas, Coke
        self.MWt = np.zeros(4)
        self.MWtElements=np.array([12.,1.,16.,14.,32.])
        self.MWt[0]=np.dot(self.MWtElements,self.zker)
        self.n=nx*nz
        self.ReactionHeat_mol=0. # Units of J/mol of kerogen
        self.ReactionHeat_gker=0. # Units of J/g of kerogen
        self.ReactionHeat_ggas=0. # Units of J/g of mobile hydrocarbon
        self.slopeLinMod=np.zeros(2)
        self.yinterLinMod=np.zeros(2)
        self.wav=np.pi*self.war**2.*self.wah
        # ========================================================================
        # ====EDIT BELOW TO ADD MORE OPTIONS TO CHANGE THE URANIUM DENSITY =======
        # ====SOLID WASTE CONTAINERS=============================================
        # ========================================================================
        if FuelSource=='Uranium':
            FuelDensity=19.0
        elif FuelSource=='Uraninite':
            FuelDensity=8.5
        # ========================================================================
        # ========================================================================
        # ========================================================================
        self.wam=FuelDensity*self.wav

        print('Total mass of uranium is', self.wam,' tonnes')

        # Computing the elementary composition of coke yielded by reactions 1 and 2
        self.zcoke1 = np.array([self.zker[0]*self.s[0,0]-self.zoil[0]*self.s[0,1]-\
            self.zgas[0]*self.s[0,2],\
            self.zker[1]*self.s[0,0]-self.zoil[1]*self.s[0,1]-self.zgas[1]*self.s[0,2],\
            self.zker[2]*self.s[0,0]-self.zoil[2]*self.s[0,1]-self.zgas[2]*self.s[0,2],\
            self.zker[3]*self.s[0,0]-self.zoil[3]*self.s[0,1]-self.zgas[3]*self.s[0,2],\
            self.zker[4]*self.s[0,0]-self.zoil[4]*self.s[0,1]-self.zgas[4]*self.s[0,2]])
        self.zcoke2 = np.array([self.zoil[0]*self.s[1,1]-self.zgas[0]*self.s[1,2],\
            self.zoil[1]*self.s[1,1]-self.zgas[1]*self.s[1,2],\
            self.zoil[2]*self.s[1,1]-self.zgas[2]*self.s[1,2],\
            self.zoil[3]*self.s[1,1]-self.zgas[3]*self.s[1,2],\
            self.zoil[4]*self.s[1,1]-self.zgas[4]*self.s[1,2]])

        self.wgastot=(self.s[0,2]/self.s[0,0]+self.s[1,2]/\
            self.s[1,1]*self.s[0,1]/self.s[0,0])*self.MWt[2]/self.MWt[0]

    def Import_BaseXML(self,fname='Base.xml'):
        ''' This method imports the base xml input file. It take as an optional
        input the name of the Base input file'''
        self.InputTree=ET.parse(fname)
        self.InputRoot=self.InputTree.getroot()

    def Update_XML(self,fname):
        ''' This method modifies the XML tree to include the new values it
        previously recalculated and creates a new input file with the  name
        given by the user in fname.'''
        # Changing the table contianing the values for the decay of radiation heat
        self.InputRoot[7][1].set('coord',' , '.join(map(str, self.Radt*86400.*365.)))
        self.InputRoot[7][1].set('value',' , '.join(map(str, self.RadH*11.25/360./self.resh)))
        self.InputRoot[6][0].set('scale',str(self.resh/self.nz))
        # print(self.InputRoot[6][0].attrib)

        # Changing the decomposition heat
        self.InputRoot[0][0].set('KerogenDecompHeat',str(self.ReactionHeat_ggas*1.0e3))

        # Changing the PT isotherm equation
        self.InputRoot[0][0].set('KerogenDecompIsothermSlope',\
            str(self.Compute_PTIsothermSlope()))
        self.InputRoot[0][0].set('KerogenDecompIsothermYinter',\
            str(self.Compute_PTIsothermYinter()))

        # Changing the Molecular wieght, prosity, initial kerogen saturation
        self.InputRoot[0][0].set('MWker',str(self.MWt[0]))
        cokemassfractionofkerogen=1.0-self.Compute_pyrolysis(1.,1.,1.,Comp='maxgas')
        print('cokemassfractionofkerogen',cokemassfractionofkerogen)
        cokevolumefractionofkerogen=cokemassfractionofkerogen*self.rhoker/self.rhocoke
        print('cokevolumefractionofkerogen',cokevolumefractionofkerogen)
        densityofgasinkerogen=(self.rhoker-cokevolumefractionofkerogen*self.rhocoke)/\
            (1.0-cokevolumefractionofkerogen)
        print('Kerogen simulated density',densityofgasinkerogen)
        self.InputRoot[0][0].set('KerogenDens',str(densityofgasinkerogen*1.0e3)) # Simulator seems to take density in units of kg/m^3
        self.InputRoot[5][2].set('value',str(1.0-self.KerVFrac/(self.por+self.KerVFrac)))
        self.InputRoot[5][6].set('value',str(self.por+self.KerVFrac))

        # Changing the Reservoir thickness, thickness of waste package, waste depth,numbers of gridblocks, end of simulation time
        self.InputRoot[1].set('zcoords',str(self.resd-self.resh)+' '+str(self.resd))
        self.InputRoot[1].set('nz',str(self.nz))
        self.InputRoot[3][0].set('point1',"0.0 0.0 "+str(self.wad-self.wah))
        self.InputRoot[3][0].set('point2',"0.0 0.0 "+str(self.wad))
        self.InputRoot[4][0].set('endtime',str(self.tend*86400.*365.))


        self.InputTree.write(fname)

    def Compute_HeatRadiation(self,tstart=0.01,nbins=10,makeplot=True):
        ''' This method computes the table representing the decaying heat by the
        nuclear decay of the nuclear waste. The inputs are as follows:
        tstart          = Starting time for the binning of discretized tabulated
                        time steps in uniform steps in log space. Units: yrs

        nbins           = Number of bins/rows used to tabulate the heat generation
                        rate through time.

        makeplot        = Boolean value which determines whther the plot is
                        generated to compare the discretized heat decay rate and
                        continuous model.'''
        # Widths of each time step
        # barw=np.exp((np.log(self.tend)-np.log(tstart))/nbins*np.arange(nbins))
        # Time at the edge of each time section
        baredges=np.logspace(np.log10(tstart),np.log10(self.tend),num=nbins+1)
        barw=baredges[1:]-baredges[:-1]
        self.Radt=baredges[:-1]
        # Center of each bar on a log scale
        bart=0.5*(baredges[1:]+baredges[:-1])
        # Heat at each row int he tabulated heat decay model
        self.RadH=self.PowerGenRate(bart)*self.wam
        # Energy produced between tstart and tend in W.yr as computed by the
        # discretized and conitnuous heat decay model
        EnergyDicretized=np.dot(barw,self.RadH)
        EnergyContinuous=self.EnergyGen(tstart,self.tend)*self.wam
        self.RadH=self.RadH*EnergyContinuous/EnergyDicretized
        EnergyDicretized=np.dot(barw,self.RadH)

        if makeplot:
            plt.figure()
            time=np.logspace(np.log(tstart),np.log(self.tend)) # in years
            plt.semilogx(time,self.PowerGenRate(time)*1.0e-3*self.wam,'r-',\
                label='t$_{cd}$='+str(self.tcd)+' yrs')
            plt.ylabel('SNF Heat generation, kW');plt.xlim((tstart,tend))
            ax=plt.gca();ybounds=ax.get_ylim();plt.ylim((0.,ybounds[1]))
            ax2=ax.twinx()
            ax2.semilogx(time,self.PowerGenRate(time)*1.0e-3,'r-')
            ax2.set_ylabel('SNF Heat generation, kW/tonne of Uranium')
            ax.set_xlabel('Time, years');ax2.set_ylim((0.,ybounds[1]/self.wam))
            bars=ax.bar(bart,self.RadH*1.0e-3,width=barw,alpha=0.5)
            bars=ax.bar(bart,self.RadH*1.0e-3,linewidth=1.,width=barw,fill=False)
            plt.tight_layout()
            plt.savefig('RadioactiveHeat_discretized.png',dpi=100)
            plt.close()

    def Compute_HeatOfReaction(self, HCC=415.e3,oil=[10,20,0,0,0],gas=[2,6,0,0,0]):
        ''' This method computes the heat consumed in Joules by the decomposition
        of 1 gram of kerogen into coke and mobile hydrocarbons which are modeled
        as methane in this simulation. The inputs are as follows:
        HCC         = Heat consumed by the rupture of 1 mol of C-C single bonds.
                    Units: J/mol'''
        self.MWt[1]=np.dot(oil,self.MWtElements)
        self.MWt[2]=np.dot(gas,self.MWtElements)
        nofCCbondsKer=[(self.zker[0]-1.)*self.s[0,0],(self.zker[0]-1.)*self.s[1,0]]
        nofCCbondsOil=[(oil[0]-1.)*self.s[0,1],(oil[0]-1.)*self.s[1,1]]
        nofCCbondsGas=[(gas[0]-1.)*self.s[0,2],(gas[0]-1.)*self.s[1,2]]
        nofCCbondsCoke=[(self.zcoke1[0]-1.)*self.s[0,3],(self.zcoke2[0]-1.)*self.s[1,3]]
        self.ReactionHeat_mol=HCC*(nofCCbondsKer[0]-nofCCbondsOil[0]-nofCCbondsGas[0]\
            -nofCCbondsCoke[0]+nofCCbondsOil[0]/nofCCbondsOil[1]*\
            (nofCCbondsOil[1]-nofCCbondsGas[1]-nofCCbondsCoke[1]))
        self.ReactionHeat_gker=self.ReactionHeat_mol/self.MWt[0]
        self.ReactionHeat_ggas=self.ReactionHeat_mol/(self.s[0,2]/self.s[0,0]+\
            self.s[1,2]/self.s[1,1]*self.s[0,1]/self.s[0,0])/self.MWt[2]

    def Compute_PTIsothermLinearEqn(self,massfraction,makeplots=True,\
            Trange=[100.,500.],Prange=[0.,30.],\
            ReactionTimes=[[1.0/365.,1./12.,0.5],[1.,5.,10.],[100.,1000.,10000.]]):
        ''' This method computes the parameters for the linear model fit to the
        isotherm on a PT chart for a fixed mass fraction of gas produced from the
        decomposition of kerogen or kerogen decomposition by-products (oil).
        of kerogen into coke and mobile hydrocarbons which are modeled as methane
        in this simulation. The inputs are as follows:
        HCC         = Heat consumed by the rupture of 1 mol of C-C single bonds.
                    Units: J/mol

        makeplots       = Boolean variable which determines whther the colormaps
                        and plots of the linear model parameters are determined.

        Trange          = Temperature range of the calculations of mass fraction
                        of gas generated. Units: degree Celsius

        Prange          = Pressure range of the calculations of mass fraction of
                        gas generated. Units: MPa

        ReactionTimes   = Reaction times for each PT isotherm computed. Units: years.
        '''

        motifs=['r','k','b']
        Time=np.logspace(-4,6.,num=1000)
        Time_seconds=Time*86400.*365.
        if type(massfraction)==list:
            nm=len(massfraction)
        elif type(massfraction)==np.ndarray:
            nm=np.size(massfraction)
        else:
            massfraction=[massfraction]
            nm=1
        fits=self.Compute_MultiPTIsotherms(Trange=Trange,Prange=Prange,times=ReactionTimes,\
                makeplots=makeplots,massfraction=massfraction)
        # print('fit',fits,'\n')

        slope=np.zeros((nm,3))
        yinter=np.zeros((nm,3))
        slope[:,0]=massfraction
        yinter[:,0]=massfraction
        if makeplots:
            plt.figure()
            ax=plt.gca()
            ax2=ax.twinx()

        for i in range(nm):
            slope[i,1:]=np.polyfit(np.log(fits[:,0]),fits[:,2*i+1],1)
            yinter[i,1:]=np.polyfit(np.log(fits[:,0]),fits[:,2*i+2],1)
            slopef=np.poly1d(slope[i,1:])
            yinterf=np.poly1d(yinter[i,1:])
            # print('S_ker='+str(massfraction[i]*100)+'%. Slope  :',slopef)
            # print('S_ker='+str(massfraction[i]*100)+'%. y-inter:',yinterf,'\n')
            if makeplots:
                ax.semilogx(Time,slopef(np.log(Time)),motifs[i]+'-',
                    label='S$_{ker}$='+str(massfraction[i]*100)+'%')
                ax2.semilogx(Time,yinterf(np.log(Time)),motifs[i]+'--')
                ax.scatter(fits[:,0],fits[:,2*i+1],facecolor='none',edgecolor=motifs[i])
                ax2.scatter(fits[:,0],fits[:,2*i+2],facecolor='none',edgecolor=motifs[i])
        self.slopeLinMod=slope[0,1:]
        self.yinterLinMod=yinter[0,1:]

        if makeplots:
            ax.arrow(fits[1,0],fits[1,1]*1.03,-fits[1,0]*0.995,0,width=0.1,
                color='k',head_length=0.0002)
            ax2.arrow(fits[-2,0],fits[-2,-1]*1.03,fits[-2,0]*50.,0,width=30.,
                color='k',head_length=50000.)
            ax.set_ylim((6,18))
            ax2.set_ylim((-5000,-1500))
            ax.set_xlim((1.0e-4,1.0e5))
            ax.legend(fancybox=False)
            ax.set_xlabel('Decomposition time, years')
            ax.set_ylabel('Slope, MPa/$\degree$C')
            ax2.set_ylabel('y-intercept, MPa')
            plt.tight_layout()
            plt.savefig('PTLinModelParam.png',dpi=100)
            plt.close()

    def Compute_PTIsothermSlope(self):
        ''' This method compute the slope of the PT isotherm for kerogen
        decomposition at the given residence time self.tr. Units: Pa/degree C '''
        return (self.slopeLinMod[0]*np.log(self.tr)+self.slopeLinMod[1])

    def Compute_PTIsothermYinter(self):
        ''' This method compute the y-intercept of the PT isotherm for kerogen
        decomposition at the given residence time self.tr. Units: Pa '''
        return (self.yinterLinMod[0]*np.log(self.tr)+self.yinterLinMod[1])

    def Compute_pyrolysis(self,P,T,t,Comp='gas'):
        ''' This method computes the concentration of the component given by the
        variable Comp after t seconds. The inputs are as follows:
        P           = Pressure. Units: Pa. Dimension must be the same as T
        T           = Temperature. Units : Kelvin. Dimensions must be the same as P
        t           = Reaction time. Units: seconds. Dimensions: Scalar if P and
                    T are arrays, otherwise, it may be an array. '''

        k1=self.A[0]*np.exp(-np.divide(self.E[0]+P*self.V,self.R*T))
        k2=self.A[1]*np.exp(-np.divide(self.E[1]+P*self.V,self.R*T))

        if Comp=='gas':
            return (self.s[0,2]*(1.0-np.exp(-k1*t))+self.s[0,1]*(self.s[1,2]/self.s[1,1])*\
                (1.+np.divide(k2*np.exp(-k1*t)-k1*np.exp(-k2*t),k1-k2)))\
                *self.MWt[2]/self.MWt[0]
        if Comp=='maxgas':
            return (self.s[0,2]+self.s[0,1]*(self.s[1,2]/self.s[1,1]))*\
                self.MWt[2]/self.MWt[0]
        elif Comp=='oil':
            return (self.s[0,1]*np.divide(k1*(np.exp(-k2*t)-np.exp(-k1*t)),k1-k2))*\
                self.MWt[1]/self.MWt[0]
        elif Comp=='generated oil':
            return (self.s[0,1]*(1.0-np.exp(-k1*t)))*self.MWt[1]/self.MWt[0]

    def Compute_MultiPTIsotherms(self,Trange=[100.,500.],Prange=[0.,30.],\
            times=[[1.0/365.,1./12.,0.5],[1.,5.,10.],[100.,1000.,10000.]],\
            massfraction=[0.5],makeplots=True):
        ''' This method computes the PT isotherms at the times given by the matrix
        times on a PT chart in the tmeperature raneg given by Trange and the
        pressure range given by Prange. The inputs are as follows.

        Trange      = Temperature range of the calculations of mass fraction of
                    gas generated. Units: degree Celsius

        Prange      = Pressure range of the calculations of mass fraction of
                    gas generated. Units: MPa

        times       = Reaction times for each PT isotherm computed. Units: years.

        massfraction= Mass fraction lines to draw the isotherms along. Must
                    contain less than or equal to 4 entries and must be a list
                    or numpy array. Units: fraction

        makeplots   = Whether the plot is generated showing the PT isotherm.
                    Units: Boolean'''
        if type(massfraction)==list:
            n=len(massfraction)
        elif type(massfraction)==np.ndarray:
            n=np.size(massfraction)
        motifs=['r-','k-','b-','g-']
        extent=[Trange[0],Trange[1],Prange[1],Prange[0]]
        Temperatures_C=np.linspace(Trange[0],Trange[1],num=1000)
        Temperatures_K=Temperatures_C+273.15
        Pressure=np.linspace(Prange[0],Prange[1],num=1000)*1.0e6 # Pa
        (TC,P)=np.meshgrid(Temperatures_C,Pressure)
        (TK,P)=np.meshgrid(Temperatures_K,Pressure)
        tarray=np.array(times)*86400.*365.
        if makeplots:
            plt.figure(figsize=(20,10))
            fig,axs=plt.subplots(3,3,sharex=True,sharey=True,figsize=(8,6))
        else:
            axs=np.zeros((3,3))
        fits=np.zeros((9,n*2+1))
        for i in range(3):
            for j in range(3):
                if i==2 and j==1:
                    showvals=[True,False]
                elif i==1 and j==0:
                    showvals=[False,True]
                else:
                    showvals=[False,False]
                if j==2:
                    showc=True
                else:
                    showc=False
                fits[i*3+j,:]=self.add_pyrolysis_colormap(tarray[i,j],TK,P,\
                    extent,motifs=motifs,massfraction=massfraction,ax=axs[i,j],\
                    showxlabel=showvals[0],showylabel=showvals[1],addcolorbar=showc,\
                    showcolorbarlabel=False,fontsize=11,showtitle=True,     \
                    makeplot=makeplots)

                if makeplots:
                    axs[i,j].set_xticks([100.,200.,300.,400.,500.])
                    axs[i,j].set_yticks([0.,10.,20.,30.])

        if makeplots:
            plt.tight_layout()
            plt.savefig('PTIsothermMultiT.png',dpi=100)
            plt.close()
        return fits


    def add_pyrolysis_colormap(self,t,T,P,extent,massfraction=[0.5],motifs=[],ax=None,\
            showxlabel=True,showylabel=True,showcolorbarlabel=True,addcolorbar=True,\
            fontsize=15,showtitle=False,makeplot=True):
        ''' This method computes a single colormap for the extent of conversion
        of kerogen to gas.

        t                   = Reaction time. Units: seconds

        T                   = Temperature 2D numpy array. Units: degree K.

        P                   = Pressure 2D numpy array. Must be the same shape as T.
                            Units: degree K

        extent              = Array contaning the x and y bounds of the PT plot.
                            The order shoud be as follows: [Tmax, Tmin, Pmin,Pmax].
                            Units: degree C and MPa.

        massfraction        = Constant mass fraction countour lines to plot.
                            Units: Mass fraction

        motifs              = Motifs to be used to plot the constant mass fraction
                            countour lines.

        ax                  = Pyplot axe object used for plotting.

        showxlabel          = Boolean variable which determine the x axis label
                            is shown on the plot. Unit: Boolean

        showylabel          = Boolean variable which determine the y axis label
                            is shown on the plot. Unit: Boolean

        showcolorbarlabel   = Boolean variable which determines whether the
                            colorbar label is shown. Unit: Boolean

        addcolorbarlabel    = Boolean variable which determines whether the
                            colorbar label is shown. Unit: Boolean

        fontsize            = Font size.

        showtitle           = Boolean variable which determines whether the
                            title is shown. Unit: Boolean

        makeplots           = Whether the plot is generated showing the PT
                            isotherm. Units: Boolean'''
        # print('T',T)
        # print('P',P)
        # print('t',t)
        CD=self.Compute_pyrolysis(P,T,t)
        # print('CD',CD)
        CD=np.divide(CD,np.max(CD))
        # print('min,max',np.min(CD),np.max(CD),'\n \n \n')
        if makeplot:
            if ax==None:
                ax=plt.gca()
            img=ax.imshow(CD,extent=extent,aspect='auto', interpolation='nearest')
            if addcolorbar:
                cbar=plt.colorbar(img,ax=ax,shrink=1.0,ticks=[0.,0.5,1.])
                cbar.ax.set_yticklabels([0.,0.5,1.])
                cbar.ax.tick_params(labelsize=fontsize)
                if showcolorbarlabel:
                    cbar.set_label('Conversion')
            ax.set_xlim((extent[0],extent[1]))
            ax.set_ylim((extent[3],extent[2]))
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)
            if showtitle:
                if t<=86400.*30.:
                    ax.set_title('t = '+str(t/86400.)+' days',fontsize=fontsize)
                elif t<=86400.*365.*0.98:
                    ax.set_title('t = '+str(t/(86400.*365./12))+' months',fontsize=fontsize)
                else:
                    ax.set_title('t = '+str(t/(86400.*365.))+' years',fontsize=fontsize)
            if showxlabel:
                ax.set_xlabel('Temperature, $\degree$C',fontsize=fontsize)
            if showylabel:
                ax.set_ylabel('Pressure, MPa',fontsize=fontsize)
        if type(massfraction)==list:
            n=len(massfraction)
        elif type(massfraction)==np.ndarray:
            n=np.size(massfraction)
        fit=np.zeros((2*n+1))
        fit[0]=t/(86400.*365.)
        for k in range(n):
            # print('fraction0',massfraction[k])
            fit[1+2*k:2*k+3]=self.add_pyrolysis_colormap_line(CD,T,P,m=motifs[k],\
                fraction=massfraction[k],ax=ax,makeplot=makeplot)
        return fit

    def add_pyrolysis_colormap_line(self, CD,T,P,m='k-',fraction=0.5,ax=None,\
            makeplot=True):
        ''' This method computes a single colormap for the extent of conversion
        of kerogen to gas.

        CD                  = Normalized mass fraction of a certain component.
                            Units: mass fraction

        T                   = Temperature 2D numpy array. Units: degree K.

        P                   = Pressure 2D numpy array. Must be the same shape as T.
                            Units: degree K

        m                   = Motif to be used to plot the constant mass fraction
                            countour line.

        fraction            = Constant mass fraction countour lines to plot.
                            Units: Mass fraction

        ax                  = Pyplot axe object used for plotting.

        makeplots           = Whether the plot is generated showing the PT
                            isotherm. Units: Boolean'''
        if makeplot:
            if ax==None:
                ax=plt.gca()
        # print('fraction',fraction)
        CD_fullyconverted=np.where(CD>=fraction,CD,0.)
        CD_fullyconverted=np.where(CD_fullyconverted<fraction+0.01,CD_fullyconverted,0.)
        fit=np.poly1d(np.polyfit(T[CD_fullyconverted!=0.]-273.15,P[CD_fullyconverted!=0.]*1.0e-6,1))
        Tarray=np.linspace(np.min(T-273.15),np.max(T-273.15))
        Parray=fit(Tarray)
        if makeplot:
            ax.plot(Tarray,Parray,m,linewidth=1.)
        return np.polyfit(T[CD_fullyconverted!=0.]-273.15,P[CD_fullyconverted!=0.]*1.0e-6,1)
    def PowerGenRate(self,t):
        ''' This method computes the rate of Power generation in W. The inputs
        are as follows:
        t               = time array. Units: years'''
        return self.Radc*np.power(t+self.tcd,self.Rada)

    def EnergyGen(self,tfrom,tto):
        ''' This method computes the energy generated in W.years. The inputs
        are as follows:
        t               = time array. Units: years'''
        return self.Radc/(self.Rada+1.)*(np.power(tto+self.tcd,self.Rada+1.)-\
            np.power(tfrom+self.tcd,self.Rada+1.))
if __name__ == '__main__':
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
    reservoirthickness=60. # m
    reservoirdepth=50. # m
    wastedepth=3.01 # m
    wastethickness=5.02 # m
    wasteradius=0.5 # m
    makeplots=False
    nx=100
    nz=60


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


    # constest=InputGenerator(tend, KerogenElementaryComp,DecompStoichiometry,\
    #     DecompFrequency,DecompActivationEnergy, DecompFreeVolume, tcooldown, \
    #     HeatDecaytsteps,KerVFracInit,PTIsothermtreaction,porosity,reservoirthickness,\
    #     wastedepth,wastethickness,nx,nz)

    # constest.Compute_HeatRadiation()
    # constest.Compute_HeatOfReaction()
    # constest.Compute_PTIsothermLinearEqn(0.1)
