#!/usr/bin/env python

"""
* ----------------------------------------------------------------------------
* A little program to analyse multispecimen paleointensity data.
* Written by Roman Leonhardt 2015
* ----------------------------------------------------------------------------
"""

from glob import glob, iglob, has_magic
from operator import add, sub, truediv
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit, leastsq
from scipy.stats import mstats, t, linregress

import random
import os
import csv
import math
import numpy as np
import getopt
import sys


SUPPORTED_FORMATS = ['MONTPELLIER','VIENNA','LEOBEN']


def XYZToFDI(XYZ):
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]
    H = np.sqrt(X**2 + Y**2) 
    F = np.sqrt(X**2 + Y**2 + Z**2)
    if not Y == 0.0:
        Dec = 180. / np.pi * np.arctan(X/Y)
    else:
        if X > 0:
            Dec = 0.0
        else:
            Dec = 180.0
    if not Z == 0.0:
        Inc = 180. / np.pi * np.arctan(H/Z) 
    else:
        Inc = 0.0
    if Inc < 0:
        F = -F
    return [F, Dec, Inc]

def FDIToXYZ(F, D, I):
    Drad = D*np.pi /180.
    Irad = I*np.pi /180.;  
    X = F * np.cos(Drad) * np.cos(Irad)
    Y = F * np.sin(Drad) * np.cos(Irad)
    Z = F * np.sin(Irad)
    return X, Y, Z

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angularDiff(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi *180.0/np.pi
    return angle *180.0/np.pi

def writeData(outpath,header,result,mode=None):

        if not mode:
            mode = 'ab'
        
        print "Saving to:", outpath
        writeheader = False
        if not os.path.isfile(outpath):
            writeheader = True
        with open(outpath, mode) as csvfile:
            mspwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if writeheader:
                mspwriter.writerow(header)
            mspwriter.writerow(result)


#(* Formula for Q calculation QDB = F; QF = Fnew; QDSC = our *) 
def QDB(m0, m1):
    return (m1 - m0)/(m0)

def QDSC(a, m0, m1, m2, m3):
    return 2.*((1 + a)*m1 - m0 - a*m3)/(2*m0 - m1 - m2)

def QFC(m0, m1, m2):
    return 2.*(m1 - m0)/(2*m0 - m1 - m2)

def QDSCmax(a, m0, m1, m2, m3, m4): 
    return 2.*((1 + a)*(m1 + np.abs(m1 - m4)) - m0 - a*m3)/(2*m0 - (m1 - np.abs(m1 - m4)) - m2)

def QDSCmin(a, m0, m1, m2, m3, m4):
    return 2.*((1 + a)*(m1 - np.abs(m1 - m4)) - m0 - a*m3)/(2*m0 - (m1 + np.abs(m1 - m4)) - m2);

#(* Uncertainty formula for function func *)(* \
#dmsum[a_,m0_,m1_,m2_,m3_] = (m1 D[QDSC[a,m0,m1,m2,m3],m1])^2 + (m2 \
#D[QDSC[a,m0,m1,m2,m3],m2])^2 + (m3 D[QDSC[a,m0,m1,m2,m3],m3])^2;
#Simplify[%] *)

def func(a, m0, m1, m2, m3): 
    return 1./(-2.*m0 + m1 + m2)**4 * 4*(a**2 * (-2.*m0 + m1 + m2)**2 * m3**2 + m2**2 * (m0 - (1 + a)*m1 + a*m3)**2 + m1**2 * (-(1 + 2. * a) * m0 + m2 + a * m2 + a * m3)**2)

def epsi(m1, m4):
    return np.abs((m1 - m4)/m1)

def Qalt2(a, m0, m1, m2, m3, m4):
    return epsi(m1, m4)**2 * func(a, m0, m1, m2, m3)

def Qds2(m0, m1, m2, m3):
    return 1./3. * ((m3 - m1)/(2*m0 - m1 - m2))**2

def Q(a, m0, m1, m2, m3, m4):
    return np.sqrt(Qalt2(a, m0, m1, m2, m3, m4) + Qds2(m0, m1, m2, m3))




def scatterfit(x,y,a=None,b=None):  
    """  
    Compute the mean deviation of the data about the linear model given if A,B  
    (y=ax+b) provided as arguments. Otherwise, compute the mean deviation about   
    the best-fit line.  
   
    x,y assumed to be Numpy arrays. a,b scalars.  
    Returns the float sd with the mean deviation.  
   
    Author: Rodrigo Nemmen  
    """  
   
    if a==None:   
        # Performs linear regression  
        a, b, r, p, err = slinregress(x,y)  
    
    # Std. deviation of an individual measurement (Bevington, eq. 6.15)  
    N=np.size(x)  
    sd=1./(N-2.)* np.sum((y-a*x-b)**2); sd=np.sqrt(sd)  
    
    return sd 

def confband(xd,yd,a,b,conf=0.95,x=None):
	"""
Calculates the confidence band of the linear regression model at the desired confidence
level, using analytical methods. The 2sigma confidence interval is 95% sure to contain 
the best-fit regression line. This is not the same as saying it will contain 95% of 
the data points.

Arguments:
- conf: desired confidence level, by default 0.95 (2 sigma)
- xd,yd: data arrays
- a,b: linear fit parameters as in y=ax+b
- x: (optional) array with x values to calculate the confidence band. If none is provided, will
  by default generate 100 points in the original x-range of the data.
  
Returns:
Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands 
corresponding to the [input] x array.

Usage:
>>> lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95)
calculates the confidence bands for the given input arrays

>>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
plots a shaded area containing the confidence band

References:
1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals
2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm

Author: Rodrigo Nemmen
v1 Dec. 2011
v2 Jun. 2012: corrected bug in computing dy
  	"""
	alpha=1.-conf	# significance
	n=xd.size	# data sample size
 
	if x==None: x=np.linspace(xd.min(),xd.max(),100)
 
	# Predicted values (best-fit model)
	y=a*x+b
 
	# Auxiliary definitions
	sd=scatterfit(xd,yd,a,b)	# Scatter of data about the model
	sxd=np.sum((xd-xd.mean())**2)
	sx=(x-xd.mean())**2	# array
 
	# Quantile of Student's t distribution for p=1-alpha/2
	q=t.ppf(1.-alpha/2.,n-2)
 
	# Confidence band
	dy=q*sd*np.sqrt( 1./n + sx/sxd )
	ucb=y+dy	# Upper confidence band
	lcb=y-dy	# Lower confidence band
 
	return lcb,ucb,x


class mspdata (object):
    """
    Needs to contain the following info:
    m0 - m4, idf, field, tstep, sample 
    """
    def __init__(self, sample=[], field=[], typ=[], temp=[], F=[], D=[], I=[], X=[], Y=[], Z=[], crit={}):
        self.crit = crit
        self.sample = [elem for elem in sample] # the specimenname
        self.typ = [elem for elem in typ] # m0,m1,etc
        self.field = [elem for elem in field] # 50
        self.temp = [elem for elem in temp]
        self.F = [elem for elem in F]
        self.D = [elem for elem in D]
        self.I = [elem for elem in I]
        self.X = [elem for elem in X]
        self.Y = [elem for elem in Y]
        self.Z = [elem for elem in Z]
        self.statslist=[]


    def recalclist(self):
        """
        Method to extract a sample specific array
        contains: sample, field, temp, m0, m1, m2, m3, m4
        """
        
        return np.asarray(ar)

    def samplelist(self):
        """
        Method to extract the amount of samples
        """
        seen = set()
        seen_add = seen.add
        return [ x for x in self.sample if not (x in seen or seen_add(x))]
        

    def setcriteria(self,criteria):
        pass

    def checkdata(self,samplelist,stats=False,outpath=False):
        """
        DESCRIPTION
           Method for basic condition checking.
           Investigated are consistency of dircetion between NRM and MSP_DSC Demag,
           fraction of the NRM, realtive differences of m1-m3 and m4-m3 steps etc 
        SPECIAL:
           if m2 is a demag step -> marked by m2*
           calculate m2 from m2*: m2* = (m1+m2)/2 -> m2 = 2*m2* - m1

        """
        mspselection = []
        for sample in samplelist:
            errorfound = False
            #print "Testing sample", sample
            ar = [[self.sample[ind], self.field[ind], self.typ[ind], self.X[ind], self.Y[ind], self.Z[ind], self.temp[ind]] for ind,elem in enumerate(self.sample) if elem == sample]

            fraction, incdiff, m3frac = 0.5,0.,0.
            # TEST 1:
            # Checking NRM fraction
            if len(ar) > 2:
                m0= [ar[0][3],ar[0][4],ar[0][5]]
                m1= [ar[1][3],ar[1][4],ar[1][5]]
                temp = ar[0][6]
                if ar[2][2].endswith('*'):
                    print "m2 is marked as zero-field demag step and not Thellier - transforming"
                    m2 = [2*ar[2][3]-ar[1][3],2*ar[2][4]-ar[1][4],2*ar[2][5]-ar[1][5]] 
                else:
                    m2= [ar[2][3],ar[2][4],ar[2][5]]
                vec = map(add,m1,m2)
                vectrm = map(sub,m1,m2)
                #print "Vectors", vec
                vec2 = [v/2. for v in vec]
                vectrm2 = [v/2. for v in vectrm]
                normnrm = np.linalg.norm(m0)
                demag = np.linalg.norm(vec2)
                fraction = (normnrm - demag)/normnrm
                nrmdiff = angularDiff(m0,vec2)
                trmdiff = angularDiff(m0,vectrm2)
                trm2diff = angularDiff(vec2,vectrm2)
                if not float(self.crit['angdiff']) > nrmdiff:
                    if not stats:
                        print "%s error: NRM differs by more than %.2f deg from demagnetized state" %(sample, float(self.crit['angdiff']))
                    if self.crit['angdiff_crit']:
                        errorfound = True

                if not float(self.crit['angdiff']) > trmdiff:
                    if not stats:
                        print "%s error: pTRM differs by more than %.2f deg from NRM" %(sample, float(self.crit['angdiff']))
                    if self.crit['angdiff_crit']:
                        errorfound = True


                if not float(self.crit['f_low']) <= fraction <= float(self.crit['f_high']):
                    if not stats:
                        print "%s error: NRM fraction outside the acceptable range from %.2f to %.2f" %(sample, float(self.crit['f_low']), float(self.crit['f_high']))
                    if self.crit['f_crit']:
                        errorfound = True

            # TEST 2:
            # Checking m1-m3 difference
            if len(ar) > 3:
                m3= [ar[3][3],ar[3][4],ar[3][5]]
                m3frac = (np.linalg.norm(m1) - np.linalg.norm(m3))/np.linalg.norm(m1)
                if m3frac < float(self.crit['m3m1threshold']):
                    if not stats:
                        print "%s warning: m3 larger then m1 - apparently alteration is affecting this check" % sample
                    if self.crit['m3m1_crit']:
                        errorfound = True
            # TEST 3:
            # Checking m4-m3 difference
            if len(ar) > 4:
                m4= [ar[4][3],ar[4][4],ar[4][5]]
                m4frac = (np.linalg.norm(m4) - np.linalg.norm(m3))/np.linalg.norm(m4)
                if m4frac < float(self.crit['m3m4threshold']):
                    if not stats:
                        print "%s warning: m3 larger then m4 - apparently alteration is affecting this check" % sample
                    if self.crit['m3m4_crit']:
                        errorfound = True

            if stats:
                self.statslist.append([sample,fraction,nrmdiff,m3frac,m4frac])

            resultlist = False
            if not errorfound:
                mspselection.append([sample,ar[0][1],XYZToFDI(m0)[0],XYZToFDI(m1)[0],XYZToFDI(m2)[0],XYZToFDI(m3)[0],XYZToFDI(m4)[0]])
                results = [sample, temp, ar[0][1], fraction, nrmdiff, trmdiff, m3frac, m4frac]
                if stats:
                    print "accepted: %20s%6.0f%6.0f%6.2f%6.2f%6.2f%6.2f%6.2f" % (sample, temp, ar[0][1], fraction, nrmdiff, trmdiff, m3frac, m4frac)
                    resultlist = ['accepted']
                    resultlist.extend(results)
            else:
                if stats:
                    print "rejected: %20s%6.0f%6.0f%6.2f%6.2f%6.2f%6.2f%6.2f" % (sample, temp, ar[0][1], fraction, nrmdiff, trmdiff, m3frac, m4frac)
                    results = [sample, temp, ar[0][1], fraction, nrmdiff, trmdiff, m3frac, m4frac]
                    resultlist = ['rejected']
                    resultlist.extend(results)
                pass
            header = ["Type" ,"sample", "temp", "field", "fraction", "nrmdiff", "trmdiff", "m3frac", "m4frac"]
            if outpath and resultlist:
                writeData(os.path.join(os.path.dirname(outpath),"myTest.txt"),header,resultlist)

        print "Remaining Samples:", len(mspselection)

        if stats:
            print "-------------------------"
            print "Providing data statistics"
            self.plotHIST(self.statslist,label=['Sample','Fraction','Angular differences', 'm1-m3', 'm4-m3'])

        return mspselection

    def limitlowerquantile(self,Qlist,prob=[0.16,0.50,0.84]):
        #prob=[0.16,0.50,0.84]
        yerr = np.asarray([el[1] for el in Qlist])
        quantiles = mstats.mquantiles(yerr,prob=prob)
        newlst = []
        for elem in Qlist:
            if elem[1] >= quantiles[0]:
                newlst.append(elem)
            else:
                newlst.append([elem[0],quantiles[0]])
        return newlst

    def calcmsp(self,mspselection,alpha=0.5,stats=False,limittolowerquantile=False):
        """
        Method to calculate all MSP lists and the median error (use lower quantile??)
        """
        MSPDBlist,MSPFClist,MSPDSClist,Qlist = [],[],[],[]
        for i in range(len(mspselection)):
            MSPDBlist.append([mspselection[i][1], QDB(mspselection[i][2], mspselection[i][3])])
            MSPFClist.append([mspselection[i][1], QFC(mspselection[i][2], mspselection[i][3], mspselection[i][4])])
            MSPDSClist.append([mspselection[i][1], QDSC(alpha, mspselection[i][2], mspselection[i][3], mspselection[i][4], mspselection[i][5])])
            Qlist.append([mspselection[i][1], Q(alpha, mspselection[i][2], mspselection[i][3], mspselection[i][4], mspselection[i][5], mspselection[i][6])])
            #print mspselection[i][1], QDB(mspselection[i][2], mspselection[i][3])
            
        if limittolowerquantile:
            Qlist = self.limitlowerquantile(Qlist)

        weights = np.asarray([el[1] for el in Qlist])

        # Get Linear fits:
        g1,i1,r1,l1,u1,xx = self.LinearModelFit(MSPDBlist)
        g2,i2,r2 = self.LinearModelFit(MSPFClist,anchored=True)
        g3,i3,r3 = self.LinearModelFit(MSPDSClist,weights=weights,anchored=True, stats=stats)

        mod1 = [g1,i1,r1,MSPDBlist,l1,u1,xx]
        mod2 = [g2,i2,r2,MSPFClist]
        mod3 = [g3,i3,r3,MSPDSClist]

        if stats:
            print "-------------------------------------"
            print "Providing alteration error statistics"
            print "Median", np.median([elem[1] for elem in Qlist])
            yerr = np.asarray([el[1] for el in Qlist])
            quantiles = mstats.mquantiles(yerr)
            #print "Quantiles:", quantiles
            self.plotHIST(Qlist,label=['Sample','Q'],Qquantile25=quantiles[0])

        return mod1, mod2, mod3, Qlist


    def plotMSP(self, lst,ax=None,symbol='o',color='b',size=25,linecolor='b',linestyle='-',linewidth=0.4,noshow=False,legend=False,results=True,slope=None, intercept=None,confidenceband=[], rsquared=None,errorbar=None,plottitle=None,scatter=True):
        """
        Method to plot a single list
        Parameters are datalist and legend
        """
        avpal = 0.0
        uncert = 0.0

        x = np.asarray([el[0] for el in lst])
        y = np.asarray([el[1] for el in lst])
        legdict = {}
        leglist=[]

        if errorbar:
            yerr = np.asarray([el[1] for el in errorbar])
            #quantiles = mstats.mquantiles(yerr)
            #print "Quantiles:", quantiles
        
        if results and slope and intercept:
            avpal = (-intercept/slope)
            restext = 'Intensity:  %.2f' % avpal
            if errorbar:
                uncert = 2.*avpal*np.sqrt( 1 / np.sum( np.divide((x/avpal),yerr)**2 ) )
                restext = restext + ' +/- %.2f' % uncert # 2 sigma error is used !!
            if rsquared:
                restext = restext + '\nR squared:  %.2f' % rsquared

        if not ax:
            fig, ax = plt.subplots(1,1, figsize=(10,4))
            try:
                plt.gca().tight_layout()
            except:
                plt.gcf().subplots_adjust(bottom=0.15)
            ax.grid(True)
            ax.axhline(0.0, linestyle='--', color='k')
            plt.xlabel('Field [mu T]')
            plt.ylabel('MSP ratio')
            plt.xlim(0.0, 100.0)
            if plottitle:
                plt.title(plottitle)

        if len(confidenceband) > 0:
            ax.fill_between(confidenceband[2],confidenceband[0],confidenceband[1],alpha=0.3, facecolor='gray')

        if scatter:
            ax.scatter(x, y, marker=symbol, c=color, s=size, label = 'data')

        if errorbar:
            ax.errorbar(x, y, yerr=yerr, fmt=None, ecolor=color, capsize=5)


        if slope and intercept:
            xs = x
            xs = np.insert(xs,0,0)
            ax.plot(xs, xs*slope + intercept, linecolor,ls=linestyle,lw=linewidth)

        if results:
            ax.text(0.2, 0.85,restext, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        #if legend:
        #    for key in legdict:
        #        leglist.append([key+':    '+legdict[key]])
        #    print leglist
        #    plt.legend([leglist], loc='upper center')

        if not noshow:
            plt.show()

        return ax, avpal, uncert

    def plotHIST(self,lst,label=[],noshow=False,leg=None,bins=20,color='black',a=0.5,Qquantile25=None):
        """
        Method to plot Histograms
        Datalist lst should contain a sample name in the first column
        """
        
        for i in range(len(lst[0])):
            if i > 0:
                val = [el[i] for el in lst]

                fig, ax = plt.subplots(1,1, figsize=(10,4))

                try:
                    plt.gca().tight_layout()
                except:
                    plt.gcf().subplots_adjust(bottom=0.15)
                plt.hist(val,bins=bins,facecolor=color, alpha=a)
                if label[i] == 'Fraction':
                    ax.axvline(self.crit['f_low'], linestyle='--', color='r')
                    ax.axvline(self.crit['f_high'], linestyle='--', color='r')
                if label[i].startswith('Angular'):
                    ax.axvline(self.crit['angdiff'], linestyle='--', color='r')
                if label[i].startswith('m1'):
                    ax.axvline(self.crit['m3m1threshold'], linestyle='--', color='r')
                if label[i].startswith('m4'):
                    ax.axvline(self.crit['m3m4threshold'], linestyle='--', color='r')
                if label[i] == 'Q' and Qquantile25:
                    ax.axvline(Qquantile25, linestyle='--', color='r')
                # Change color !!!!
                plt.ylabel("N")
                try:
                    plt.xlabel(label[i])
                    if label[i] == 'Q':
                        plt.title("Uncertainty distribution")
                    if label[i] == 'Fraction':
                        plt.title("NRM fraction distribution")
                    if label[i].startswith('Angular'):
                        plt.title("Distribution of angular deviations between NRM and Demag")
                    if label[i].startswith('m1'):
                        plt.title("Distribution differences between m1 and m3")
                    if label[i].startswith('m4'):
                        plt.title("Distribution differences between m4 and m3")
                    if label[i].startswith('Inten'):
                        plt.title("Jackknife distribution")
                except:
                    pass


        if not noshow:
            plt.show()


    def LinearModelFit(self, lst, anchored=False, weights=[], stats=False):

        '''Fits a linear fit of the form mx+b of mx-1 (if anchored) to the data'''
        x = np.asarray([el[0] for el in lst])
        y = np.asarray([el[1] for el in lst])

        if len(weights) == 0:
            weights = np.asarray([1.0]*len(y))
        else:
            if not len(weights) == len(y):
                print "Take care: weigths do not fit data length"
                weights = np.asarray([1.0]*len(y))
            weights = np.asarray(weights)
           
        if not anchored:
            fitfunc = lambda params, x: params[0] * x + params[1] 
            init_a = 0.5                     #find initial value for a (gradient)
            init_b = min(y)                  #find initial value for a (gradient)
            init_p = np.array((init_a,init_b))  #bundle initial values in initial parameters
        else:
            fitfunc = lambda params, x: params[0] * x - 1.0
            init_a = 0.5                  #find initial value for a (gradient)
            init_p = np.array((init_a))  #bundle initial values in initial parameters

        errfunc = lambda p, x, y: (fitfunc(p, x) - y)/weights

        #calculate best fitting parameters (i.e. m and b) using the error function
        p1, success,infodict,mesg,ier = leastsq(errfunc, init_p.copy(), args = (x, y),full_output=True)
        f = fitfunc(p1, x)          #create a fit with those parameters

        def residuals(a,x,y):
            return y-f(x,a)

        ss_err=(infodict['fvec']**2).sum()
        ss_tot=((1/weights*(y-y.mean()))**2).sum()
        rsquared=1-(ss_err/ss_tot)

        gradient = p1[0]
        intercept = -1.0
        if len(p1) > 1:
            intercept = p1[1]

        if not anchored:
            lcb,ucb,xx=confband(x,y,gradient,intercept,conf=0.95)
            return gradient, intercept, rsquared, lcb, ucb, xx

        if stats:
            print "Anchored:", anchored
            print "Gradient and intercept", gradient, intercept
            print "Zerocrossing at", -intercept/gradient
            print "R-squared", rsquared

        return gradient, intercept, rsquared


    def JackKnifeMSP(self,mspselection,alpha=0.5,percentage=20, maxnum=5,limittolowerquantile=False):
        """
        Method to drop randomly up to 20 percent from the collection
        and analyse the reduced data set
        """
        WData = []
        pallist = []

        def makeDrippingBucket(lst):
            bucket = lst
            if len(bucket) == 0:
                return []
            else:
                random_index = random.randrange(0,len(bucket))
                del bucket[random_index]
                return bucket

        print "Running JackKnife statistics with alpha:", alpha

        # Firstly join weights and Data
        for i in range(len(mspselection)):
            WData.append([mspselection[i][1], QDSC(alpha, mspselection[i][2], mspselection[i][3], mspselection[i][4], mspselection[i][5]),Q(alpha, mspselection[i][2], mspselection[i][3], mspselection[i][4], mspselection[i][5], mspselection[i][6])])
        dl = len(WData)
        projectedminimum = dl - np.round(dl*percentage/100)
        for i in range(dl*maxnum):
            buck = [el for el in WData]
            # Drop random elements so that the amount is equal or larger the projected minimum:
            # Use the following amount of specimens
            randomnum = random.random()
            projectedsamples = np.round(projectedminimum + ((dl - 1) - projectedminimum)*randomnum)
            while len(buck) >= projectedsamples:
                buck = makeDrippingBucket(buck)
            # Split up data and get MSPDSC and Qlist again
            mspdsc = [[el[0],el[1]] for el in buck]
            Qlist = [[el[0],el[2]] for el in buck]
            if limittolowerquantile:
                Qlist = self.limitlowerquantile(Qlist)
            qlst = [el[1] for el in Qlist]
            #print len(mspdsc), mspdsc[0]
            g3,i3,r3 = self.LinearModelFit(mspdsc,anchored=True,weights=qlst)
            pallist.append([r3,-i3/g3])

        pals = [el[1] for el in pallist]

        self.plotHIST(pallist,label=['Rsquared','Intensity'],color='black',a=0.5)

        # get median and quantiles from palintensity distribution
        #  which would correspond to a sigma error in case of normal distributions
        quantiles = mstats.mquantiles(pals,prob=[0.16,0.50,0.84])

        return quantiles[1], (quantiles[2]-quantiles[0])


    def getAlpha(self,mspselection,limittolowerquantile=False):
        """
        Method to estimate alpha from the data set
        """
        MSPDSClist = []
        Qlist = []
        rlst = []
        minalpha = 0.0
        maxalpha = 1.0
        alpharange = np.arange(minalpha, maxalpha, 0.01)
        for alpha in alpharange:
            for i in range(len(mspselection)):
                MSPDSClist.append([mspselection[i][1], QDSC(alpha, mspselection[i][2], mspselection[i][3], mspselection[i][4], mspselection[i][5])])
                Qlist.append([mspselection[i][1], Q(alpha, mspselection[i][2], mspselection[i][3], mspselection[i][4], mspselection[i][5], mspselection[i][6])])

            if limittolowerquantile:
                Qlist = self.limitlowerquantile(Qlist)
            weights = np.asarray([el[1] for el in Qlist])
            g3,i3,r3 = self.LinearModelFit(MSPDSClist,anchored=True,weights=weights)
            rlst.append(r3)

        bestalpha = alpharange[rlst.index(max(rlst))]

        fig, ax = plt.subplots(1,1, figsize=(10,4))
        try:
            plt.gca().tight_layout()
        except:
            plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylabel("R squared")
        plt.xlabel("alpha")
        #plt.title("R squared in dependecy of alpha")
        ax.plot(alpharange,rlst)
        ax.axvline(bestalpha, linestyle='--', color='r')
        txt = "alpha=%.2f" % bestalpha
        ax.text(0.85, 0.85, txt, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        plt.show()

        return bestalpha


# #####################################################################################
# ############### Library part beginning below ########################################
# #####################################################################################

def isVienna(filename):
    """
    My File type
     gibt auch isMontpellier
    """
    try:
        temp = open(filename, 'rt').readline()
    except:
        return False
    templst = temp.split()
    print "Linelength", len(templst)
    if not len(templst) == 7:
        return False
    if not 'Specimen' in templst:
        return False
    if not 'Step' in templst:
        return False
    return True

def getVienna(filename, **kwargs):
    """
    Looks like:
    Specimen	Step	M	D	I	T	Field
    SE1K1Y13	m0	50.9	34.4	86.0	350	15.5
    SE1K1Y13	m1	31.27	335.1	87.4	350	15.5
    """
    print "Loading Vienna/Leoben file type"

    msp = mspdata()
    getdat = True
    meta = {}

    fh = open(filename, 'rt')
    for line in fh:
        if getdat:
            cols = line.split()
            if line.startswith('#'):
                pass
            elif line.startswith('Specimen'): # found header
                pass
            elif len(cols) == 7:
                sample = cols[0]
                field = float(cols[6])
                msp.sample.append(sample)
                msp.field.append(field)
                msp.temp.append(float(cols[5]))
                msp.typ.append(cols[1])
                F = float(cols[2])
                D = float(cols[3])
                I = float(cols[4])
                msp.D.append(D)
                msp.I.append(I)
                msp.F.append(F)
                X,Y,Z = FDIToXYZ(F,D,I)
                msp.X.append(X)
                msp.Y.append(Y)
                msp.Z.append(Z)
            else:
                continue

    msp.meta = meta

    return msp

def isLeoben(filename):
    """
    Testing old Leoben fileformat
    """
    try:
        temp = open(filename, 'rt').readline()
    except:
        return False
    templst = temp.split()
    if not 'm0' in templst:
        return False
    if not 'm1' in templst:
        return False
    return True

def getLeoben(filename, **kwargs):
    """
    Reading Leoben fileformat
    Looks like:
    Name	Field	m0	m1	m2	m3	m4
    BEH1 15	0.136292236	0.04051962	-0.037854078	0.042047812	0.041871967
    BEH2 30	0.101960232	0.062783773	-0.054289023	0.063998212	0.064312668
    """

    temperature = kwargs.get("temperature")

    print "Loading old Leoben file type:"
    print "------------------------------------------"
    print "please provide the temperature step"
    print "------------------------------------------"

    msp = mspdata()
    getdat = True
    meta = {}
    samplename = "Dummy"
    if not temperature:
        temperature = 400.0
    
    fh = open(filename, 'rt')
    for idx,line in enumerate(fh):
        if getdat:
            cols = line.split()
            if line.startswith('Name') and 'm0' in cols:
                vals = len(cols)-2
                header = cols
                pass
            elif len(cols) == 7:
                sample = cols[0]
                if sample in msp.sample:
                    print "Specimen already existing - adding line number to specimen name"
                    sample = sample+str(idx)
                field = float(cols[1])
                for i in range(0,vals):
                    m = float(cols[i+2])
                    msp.sample.append(sample)
                    msp.field.append(field)
                    msp.typ.append(header[i+2])
                    msp.temp.append(temperature)
                    F = m
                    D = 0.0
                    I = 90.0
                    msp.D.append(D)
                    msp.I.append(I)
                    msp.F.append(F)
                    X,Y,Z = FDIToXYZ(F,D,I)
                    msp.X.append(X)
                    msp.Y.append(Y)
                    msp.Z.append(Z)
            else:
                continue

    msp.meta = meta

    return msp



def isMontpellier(filename):
    """
    Testing Pierres fileformat
    """
    try:
        temp = open(filename, 'rt').readline()
    except:
        return False
    templst = temp.split()
    if not len(templst)==17:
        return False
    return True

def getMontpellier(filename, **kwargs):
    """
    Reading Pierres fileformat
    Looks like:
    10.0 1.009e-2 110.0  61.0 5.986e-3 113.1  60.6  4.242e-3 116.2  61.3 5.275e-3 113.4  60.4 5.240e-3 116.5  61.7  068D
    """

    temperature = kwargs.get("temperature")

    print "Loading Pierres file type:"
    print "------------------------------------------"
    print "please provide the temperature step"
    print "------------------------------------------"

    msp = mspdata()
    getdat = True
    meta = {}
    samplename = "Dummy"

    if not temperature:
        temperature = 400.0
    
    fh = open(filename, 'rt')
    for line in fh:
        if getdat:
            cols = line.split()
            if line.startswith('#'):
                pass
            elif len(cols) == 17:
                sample = cols[16]
                field = float(cols[0])
                for i in range(int((len(cols)-2)/3.)):
                    msp.sample.append(sample)
                    msp.field.append(field)
                    msp.temp.append(temperature)
                    msp.typ.append("m"+str(i))
                    F = float(cols[1+3*i])
                    D = float(cols[2+3*i])
                    I = float(cols[3+3*i])
                    msp.D.append(D)
                    msp.I.append(I)
                    msp.F.append(F)
                    X,Y,Z = FDIToXYZ(F,D,I)
                    msp.X.append(X)
                    msp.Y.append(Y)
                    msp.Z.append(Z)
            else:
                continue

    msp.meta = meta

    return msp

def putPLAIN(path,mspresults):
    pass

def readFormat(filename, format_type, **kwargs):
    data = []
    if (format_type == "MONTPELLIER"):
        return getMontpellier(filename, **kwargs)
    elif (format_type == "VIENNA"):
        return getVienna(filename, **kwargs)
    elif (format_type == "LEOBEN"):
        return getLeoben(filename, **kwargs)
    else:
        return 

def isFormat(filename, format_type):
    if (format_type == "MONTPELLIER"):
        if (isMontpellier(filename)):
            return True
    elif (format_type == "VIENNA"):
        if (isVienna(filename)):
            return True
    elif (format_type == "LEOBEN"):
        if (isLeoben(filename)):
            return True
    else:
        return False

def _read(filename, dataformat, **kwargs):
    data = []
    meta = {}
    rock = mspdata()
    format_type = None
    if not dataformat:
        # auto detect format - go through all known formats in given sort order
        for format_type in SUPPORTED_FORMATS:
            # check format
            if isFormat(filename, format_type):
                print "Found data of format", format_type
                break
    else:
        # format given via argument
        dataformat = dataformat.upper()
        try:
            formats = [el for el in SUPPORTED_FORMATS if el == dataformat]
            format_type = formats[0]
        except IndexError:
            msg = "Format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (dataformat, ', '.join(PYMAG_SUPPORTED_FORMATS)))

    rock = readFormat(filename, format_type, **kwargs)

    return rock

def _write(path, mspresults, dataformat=None, **kwargs):

    if not dataformat:
        dataformat = 'PLAIN'

    putPLAIN(path,mspresults)


# #####################################################################################
# ############### Main functions and analysis  ########################################
# #####################################################################################


def mspanalysis(infile=None,inputformat=None,outpath=None,alpha=0.5,criteria=None,stats=False, limittolowerquantile=False, estimatealpha=False, jackknife=False, savetofile=False, **kwargs):

    median, quantile1684 = 0.0, 0.0
    bestalpha = 0.0

    #temperature = kwargs.get('temperature')

    msp = mspdata()

    if not outpath:
        outpath = os.path.dirname(infile)
        #outpath = os.path.join(outpath, "msp-results.txt")
    else:
        if os.path.isdir(outpath):
            outpath = os.path.join(outpath, "msp-results.txt")
        savetofile = True

    if os.path.isfile(infile):
        print "Reading file", infile
        msp = _read(infile, inputformat, **kwargs)
    else:
        print "Could not open file"

    if criteria:
        #print "Default criteria", msp.crit
        msp.crit = criteria
        print "Using criteria", msp.crit

    slist = msp.samplelist()
    print "Found Samples:", len(slist)

    if len(slist) < 1:
        print "Check your file"
        sys.exit()

    # Preselect data using criteria
    # ############################################
    mspselection = msp.checkdata(slist,stats=stats,outpath=outpath)

    print "starting analysis:", alpha, limittolowerquantile
    # Get the lists and median alteration error
    # ############################################
    mod1,mod2,mod3,qlst = msp.calcmsp(mspselection,alpha=alpha,stats=stats,limittolowerquantile=limittolowerquantile)

    # Get slopes for predefined alpha
    a1,a2,a3,aq = msp.calcmsp(mspselection,alpha=0.2,stats=False,limittolowerquantile=limittolowerquantile)
    b1,b2,b3,bq = msp.calcmsp(mspselection,alpha=0.8,stats=False,limittolowerquantile=limittolowerquantile)

    # Plotting DB, Field corrected and DSC diagramms with default alpha (or given alpha)
    # ############################################
    if len(mod1) > 0:
        msp.plotMSP(mod1[3],legend=False,noshow=True, plottitle="MSP DB", slope=mod1[0], intercept=mod1[1],rsquared=mod1[2],confidenceband=[mod1[4],mod1[5],mod1[6]],symbol='.',linecolor='m')
    if len(mod2) > 0:
        ax, pal, u = msp.plotMSP(mod1[3],results=False,legend=False,noshow=True,slope=mod1[0], intercept=mod1[1],rsquared=mod1[2],plottitle="MSP FC",symbol='.',linecolor='m')
        msp.plotMSP(mod2[3],ax=ax,legend=False,noshow=True,slope=mod2[0],intercept=mod2[1],rsquared=mod2[2])
    if len(mod2) > 0:
        ax, pal, u = msp.plotMSP(mod2[3],results=False,legend=False,noshow=True,slope=mod2[0], intercept=mod2[1],rsquared=mod2[2], plottitle="MSP DSC", color='b', symbol='.', linecolor='b', linestyle='--', linewidth=0.2)
        msp.plotMSP(a3[3],ax=ax,results=False,legend=False,noshow=True,slope=a3[0], intercept=a3[1],rsquared=a3[2],scatter=False,linecolor='r',linestyle='-.',linewidth=0.4)
        msp.plotMSP(b3[3],ax=ax,results=False,legend=False,noshow=True,slope=b3[0], intercept=b3[1],rsquared=b3[2],scatter=False,linecolor='g',linestyle='-.',linewidth=0.4)
        ax, avpal, uncert = msp.plotMSP(mod3[3],ax=ax,legend=False,results=True,slope=mod3[0],intercept=mod3[1],rsquared=mod3[2], color='black', linecolor='black',linewidth=1.0, errorbar=qlst)

    # Checking the differences because of alpha variations and the uncertainty
    # ############################################
    diffsofestimates = ((-a3[1]/a3[0] - -b3[1]/b3[0])/(-mod3[1]/mod3[0]))/(uncert/(-mod3[1]/mod3[0]))
    print "Normalized Differences between alpha 0.2 and alpha 0.8; and relative uncertainty:", ((-a3[1]/a3[0] - -b3[1]/b3[0])/(-mod3[1]/mod3[0]))/(uncert/(-mod3[1]/mod3[0]))

    # If alpha variations larger than uncertainty force alpha calculation
    # ############################################

    if estimatealpha or diffsofestimates > 1:
        if estimatealpha:
            print "----------------------------------------------------------------------"
            print "Getting alpha"
            print "----------------------------------------------------------------------"
        else:
            print "----------------------------------------------------------------------"
            print "Forcing alpha determination"
            print "----------------------------------------------------------------------"
        bestalpha = msp.getAlpha(mspselection,limittolowerquantile=limittolowerquantile)
        if 0.1 < bestalpha < 0.9:
            # Get the lists and median alteration error
            mod1,mod2,mod3,qlst = msp.calcmsp(mspselection,alpha=bestalpha,stats=stats,limittolowerquantile=limittolowerquantile)
            ax, pal, u = msp.plotMSP(mod2[3],results=False,legend=False,noshow=True,slope=mod2[0], intercept=mod2[1],rsquared=mod2[2],symbol='.',linecolor='b',linestyle='--',linewidth=0.1)
            msp.plotMSP(a3[3],ax=ax,results=False,legend=False,noshow=True,slope=a3[0], intercept=a3[1],rsquared=a3[2],scatter=False,linecolor='r')
            msp.plotMSP(b3[3],ax=ax,results=False,legend=False,noshow=True,slope=b3[0], intercept=b3[1],rsquared=b3[2],scatter=False,linecolor='g')
            ax, avpal, uncert = msp.plotMSP(mod3[3],ax=ax,legend=False,results=True,slope=mod3[0],intercept=mod3[1],rsquared=mod3[2], color='black', linecolor='black',linewidth=1.0, plottitle="MSP(DSC)", errorbar=qlst)
        else:
            print "Automatically determined alpha insignificant"
            bestalpha = 0.5
    else:
        bestalpha = alpha

    # Run jackknife - maybe use that as default for Q
    # ############################################

    if jackknife:
        # JackKnife
        print "Testing JackKnife"
        median, quantile1684 = msp.JackKnifeMSP(mspselection, alpha=bestalpha, percentage=20, maxnum=100,  limittolowerquantile=limittolowerquantile)

        QDSC = (1+mod3[2])/(quantile1684/median)  # replace 1 by a N dependent value
        print "----------------------------------------"
        print "Quality parameter Q(DSC): ", QDSC
        print "----------------------------------------"
        if QDSC > 20:
            print "Acceptable (Q DSC > 20)"
            print "Intensity (default), 2 sigma, Median, Quantile, Q", avpal, uncert, median, quantile1684, QDSC
        else:
            print "Reject  !!!!!!!!!!"
            print "Intensity (default), 2 sigma, Median, Quantile, Q", avpal, uncert, median, quantile1684, QDSC
        print "----------------------------------------"

    if savetofile:
        header = ["Filename", "BestAlpha", "MeanPalint", "Uncert", "MedianPalint", "Quantile1684", "Quality"] 
        resultlist = [infile, bestalpha, avpal, uncert, median, quantile1684, QDSC]
        writeData(outpath,header,resultlist)
        """
        print "Saving to:", outpath
        writeheader = False
        if not os.path.isfile(outpath):
            writeheader = True
        with open(outpath, 'ab') as csvfile:
            mspwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if writeheader:
                mspwriter.writerow(header)
            mspwriter.writerow(resultlist)
        # save results to a file
        pass
        """

def main(argv):
    source = ''		
    destination = None
    inputformat = None
    outputformat = 'RMA'
    estimatealpha=False
    ltlq=False
    stats = False
    jackknife = False
    alpha = 0.5
    temperature = None
    crit={}

    # Setting default criteria
    crit['f_low'] = '0.2'          # lower fraction limit
    crit['f_high'] = '1.0'         # upper fraction limit
    crit['angdiff'] = '10.0'       # maximum angular difference between NRM and demag step
    crit['zonly'] = False
    crit['m3m1threshold'] = '-0.01'
    crit['m3m4threshold'] = '-0.01'
    crit['f_crit'] = True	  # if True than remove sample, if False just send warning
    crit['angdiff_crit'] = True
    crit['m3m1_crit'] = False
    crit['m3m4_crit'] = False

    try:
        opts, args = getopt.getopt(argv,"hs:i:d:c:a:t:qelj",["source=","inputformat=","destination=","criteria=","alpha=","statistics=",])
    except getopt.GetoptError:
        print 'Usage:'
        print 'mspanalysis.py -s <source> -i <inputformat> -d <destination>  -a <alpha> -c <criteria> -q <statistics> -e <estimatealpha> -l <limitweight> -j <jackknife> -t <temperature>'
        print 'use mspanalysis.py -h for details.'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print '-------------------------------------'
            print 'Description:'
            print 'mspanalysis.py reads Multispecimen data and analyzes it' 
            print ''
            print '-------------------------------------'
            print 'Usage:'
            print 'mspanalysis.py -s <source> -i <inputformat> -d <destination> -a <alpha> -c <criteria> -q <statistics> -e <estimatealpha> -l <limitweight> -j <jackknife> -t <temperature>'
            print '-------------------------------------'
            print 'Options:'
            print '-s (required) : provide a sourcefile'
            print '                  A file contains individual results from ONE site.'
            print '-i            : specify input format - if not provided format will be'
            print '                  automatically determined. Supported typs are: VIENNA,'
            print '                  MONTPELLIER, LEOBEN '
            print '-d            : destination file to append results.'
            print '-c            : criteria -- Available inputs are:'
            print '                  -c alpha=0.5,inct=80,f_low=0.2,f_high=1.0,etc'
            print '-a            : provide alpha value - default is 0.5'
            print '-q            : plain option - plot some statistics like fraction'
            print '                  distribution etc.'
            print '-e            : plain option - estimate alpha'
            print '-l            : limit lower uncertainty range to lower quantile of all data.'
            print '                  This avoids strong weighting of single special data points.'
            print '-j            : Use Jack Knife statistics.'
            print '-t            : temperature at which the experiment was performed.'
            print '                  Will be appended to the result file. Please provide for'
            print '                  Leoben and Montpellier file types'
            print '-------------------------------------'
            print 'Examples:'
            print 'MSPTool.py -s "/pyth/to/my/multispecimen.dat"'
            print 'MSPTool.py -s "/pyth/to/my/multispecimen.dat -l -q -j -c f_low=0.2"'
            sys.exit()
        elif opt in ("-s", "--source"):
            source = arg
        elif opt in ("-d", "--destination"):
            destination = arg
        elif opt in ("-i", "--inputformat"):
            inputformat=arg
        elif opt in ("-c", "--criteria"):
            critstring = arg
            critlist = critstring.split(',')
            print "Criteria list", critlist
            # Eventually add here an assert list
            try:
                for elem in critlist:
                    critelem = elem.strip().split('=')
                    print "Element looks like", critelem
                    if len(critelem)==2:
                        key = critelem[0].strip()
                        if critelem[1].strip() == 'True':
                            val=True
                        elif critelem[1].strip() == 'False':
                            val=False
                        else:                      
                            val = critelem[1].strip()
                        crit[key] = val
            except:
                print 'Could not intrepret your criteria list!'
                print '-- check mspanalysis.py -h for options and requirements'
        elif opt in ("-q", "--statistics"):
            stats=True
        elif opt in ("-e", "--estimatealpha"):
            estimatealpha=True
        elif opt in ("-a", "--alpha"):
            alpha=float(arg)
        elif opt in ("-l", "--limitweight"):
            ltlq=True
        elif opt in ("-j", "--jackknife"):
            jackknife=True
        elif opt in ("-t", "--temperature"):
            temperature=float(arg)

    if source == '' or not os.path.isfile(source):
        print 'Specify a proper data source: -s /path/to/my/data.dat !'
        print '-- check mspanalysis.py -h for more options and requirements'
        sys.exit()

    mspanalysis(infile=source,inputformat=inputformat,outpath=destination,temperature=temperature,alpha=alpha,criteria=crit,stats=stats,limittolowerquantile=ltlq,estimatealpha=estimatealpha,jackknife=jackknife)
    

if __name__ == "__main__":
   main(sys.argv[1:])


