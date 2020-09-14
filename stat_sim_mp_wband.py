import matplotlib.pyplot as plt
import numpy as np
import time
import sys,glob
import scipy as sc
from astropy.table import Table,Column,join
from astropy.io import fits
import broadmag_sim
import wband_sim
import fnmatch
from multiprocessing import Pool
from random import choices
from mpl_toolkits.axes_grid1 import make_axes_locatable

# FUNCTIONS TO ADD SPECTRAL TYPES OR MASSES TO STAR TABLES, BASED ON TABLE 5 OF MAMAJEK
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def closest(lst,K):
	lst = np.asarray(lst)	
	idx = (np.abs(lst-K)).argmin()
	return idx,lst[idx]

def add_spt(tab,matching,intable,pop):
	if 'y' in pop:
		param_table = Table.read('teff_spt_table_l1.txt',format='ascii')
	else:
		param_table = Table.read('teff_spt_table.txt',format='ascii')
	list_to_match = param_table[str(matching)]
	spt_col,spt_ind_col = [],[]
	for i in range(len(tab)):
		spt_ind,mass = closest(list_to_match,tab[str(intable)][i])
		spt_col.append(param_table['SpT'][spt_ind])
		if 'y' in pop:	
			spt_ind_col.append(spt_ind-79)
		else:
			spt_ind_col.append(spt_ind)
	spectraltype = Column(spt_col,name='SpT')
	spectraltypeindex = Column(spt_ind_col, name = 'SpTind')
	tab.add_columns([spectraltype,spectraltypeindex])
	return tab


# FUNCTION TO PLOT HISTOGRAM OF SPECTRAL TYPES FROM OBJECTS IN BS MODEL
# - - - - - - - - - - - - - - - - - - - - - - - -
def histogram():

    filename = sys.argv[1]
    model = Table.read(filename,format='ascii')
    print(len(model))

	#add spectral type to table if needed:
    #model = add_spt(model)
    #model.write('ss_trilegal_final_spt.txt',format='ascii')	

    teff = model['logTe']
    bins = np.linspace(min(teff),max(teff),20)

    plt.figure()
    plt.hist(teff,bins,facecolor='paleturquoise',edgecolor='k')
    plt.vlines(3.57,0,1200,color='k',linestyle='dashed')
    plt.vlines(3.352,0,1200,color='blue',linestyle='dashed')
    #plt.xticks((1,2,3,4,5,6,7),('O','B','A','F','G','K','M'))
    plt.xlabel('Teff')
    plt.ylabel('N')
    plt.ylim(0,1000)
    #plt.title('Distribution of Spectral Types')

    plt.show()
    
    indM,indL,indT= [],[],[]
    for i in range(len(model)):
        if teff[i] <= 3.57:
            indM.append(i)
        if teff[i] <= 3.325:
            indL.append(i)
        if teff[i] <= 3.1:
            indT.append(i)

    print('Percentage of M or later = '+str((len(indM)/len(model))*100))
    print('Percentage of L or later = '+str((len(indL)/len(model))*100))
    print('Percentage of T or later = '+str((len(indT)/len(model))*100)) 

    indO,indB,indA= [],[],[]
    for i in range(len(model)):
        if teff[i] >= 4.505:
            indO.append(i)
        if teff[i] >= 3.872 and teff[i] <= 3.987:
            indA.append(i)
        if teff[i] >= 4.017 and teff[i] <= 4.498:
            indB.append(i)

    print(len(indO),len(indB),len(indA))

    return model



# FUNCTION TO CALCULATE Q INDEX OF MODEL OBJECTS USING WBAND,J,H CFHT PHOTOMETRY
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - -- - - - - - - 
def find_wband_cols(av,model,filetype):

	#IF FINDING Q FOR MODEL OBJECTS, USE FILETYPE = BS
    if 'bs' in filetype:
        num_types = ['B','A','F','G','K','M']
        spt = model['Typ']
        bs_d = model['Dist']*1000.0

		#DONT HAVE DISTANCES FOR A OR B STANDARDS, OR K8 & K9
        if spt >= 4 and spt <8:
            broad_spt = num_types[int(spt)-2]
            sub_spt = str(spt-int(spt))[2:]
            str_spt = str(broad_spt) + str(sub_spt[0])
            if 'K8' in str_spt or 'K9' in str_spt:
                filter_mags = [np.nan,np.nan,np.nan,np.nan,np.nan]
                h_cfht_app = np.nan
            else:
				# returns J(CFHT), W(CFHT), H(CFHT), H(2MASS) 
                filter_mags = wband_sim.calc_mags(str_spt,av,filetype)
                h_2mass_app = filter_mags[4]
                h_cfht_app = calc_h(h_2mass_app,str_spt,av,bs_d) 
        else: 
            filter_mags = [np.nan,np.nan,np.nan,np.nan,np.nan]
            h_cfht_app = np.nan

    elif 'tl' in filetype:
        # if using an actual trilegal model uncomment these lines
        tl_dm = model['m-M0']		
        tl_d = 10.*10.**(tl_dm/5.)
        #tl_d = model['Dist(pc)']

		#remove subtype 
        spt = model['SpT'][0:2]
        #can't find colours for L or T objects - no standards!
        if 'K6' in spt or 'K8' in spt or 'K9' in spt or 'O' in spt or 'B' in spt or 'A' in spt:
            filter_mags = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            h_cfht_app = np.nan
        else:
		# M standards avaliable for 0.5 subtypes, but only for Ms
            if 'M0' in spt or 'M1' in spt or 'M2' in spt or 'M3' in spt or 'M4' in spt or 'M6' in spt or 'M9' in spt:
                spt = model['SpT'][0:4]
                print(spt)
                filter_mags = wband_sim.calc_mags(spt,av,filetype)
                h_2mass_app = filter_mags[4]
                h_cfht_app = calc_h(h_2mass_app,spt,av,tl_d,filetype,filter_mags[5])
            else:
                print(spt)
                filter_mags = wband_sim.calc_mags(spt,av,filetype)
                h_2mass_app = filter_mags[4]
                h_cfht_app = calc_h(h_2mass_app,spt,av,tl_d,filetype,filter_mags[5]) 

    elif 'l17' in filetype or 'realfile' in filetype:
        
        spt = model 
        #for 'l17' uncomment this!
        if 'l17' in filetype:
            spts = model['SpT']
            c = spts.split('V')
            spt = c[0]
        
        #y_d = model['Dist(pc)']
        if 'L1' in spt or 'M0' in spt or 'M1' in spt:
            filter_mags = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            h_cfht_app = np.nan
        else:
            filter_mags = wband_sim.calc_mags(spt,av,filetype)            
            h_2mass_app = 0.0
            #If distance comes from a distribution, change 436 to y_d and uncomment line above
            h_cfht_app = calc_h(h_2mass_app,spt,av,436.,filetype) 

    #IF FINDING Q FOR 'REAL' OBJECTS IE NOT USING STANDARD SPECTRA, USE FILETYPE = REALFILE
    else:
        filter_mags = wband_sim.calc_mags(model,av,filetype)

    #Ks-W, Ks-J, Ks-H, J-W, H-W
    #return filter_mags
    return filter_mags[3]-filter_mags[1],filter_mags[3]-filter_mags[0],filter_mags[3]-filter_mags[2],filter_mags[0]-filter_mags[1],filter_mags[2]-filter_mags[1],h_cfht_app



# FUNCTION TO CALCULATE APPARENT MAGNITUDES OF BS MODEL OBJECTS AT THEIR SIMULATED DISTANCES
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
def calc_h(h2,spt,av,bs_d,typ,name):

    if 'l17' in typ or 'realfile' in typ:
        t = Table.read('simbad_l17_dist.txt',format='ascii',guess=False)
        for i in range(len(t)):
            if spt in it['SpT'][i]:
            #chosen_std = t['SpT'][i]
                dist = t['Dist (pc)'][i]
                h2 = t['Happ'][i]
    else:
        t = Table.read('simbad_dis.txt',format='ascii',guess=False)
        for i in range(len(t)):
            if t['Name'][i] in name:
                chosen_std = t['SpT'][i]
                dist = t['Dist(pc)'][i]

    habs = abs_mag(h2,dist)
    happ = app_mag(habs,bs_d,-av)

    return happ


# FUNCTION TO CALCULATE INTRINISC ABS AND APP MAGS, AND CUSTOM FILTER MAGS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def abs_mag(m,d):
    return m - 5*np.log10(d) + 5

def app_mag(m,d,a):
    return -5 + 5*np.log10(d) + m + 0.19*a

def filter_mags(model,col):

    k_ap = model['K']
    dist = model['Dist']*1000

    ap = k_ap - col
    ab = abs_mag(ap,dist)

    return ap,ab


# FUNCTION: MULTIPROCESSING LOOP - SPLIT BS TABLE INTO 4 PARTS AND CALCULATE Q INDEX FASTER FOR FULL TABLE
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def mp_loop_single_wband_table(model):

    q_cols,H_mags,col1,col2 = [],[],[],[]
    #typ = str(sys.argv[1])
    typ = 'tl'
    for i in range(len(model)):
        # CHANGE FIRST ARGUMENT TO -MODEL[I]['AV'] IF USING DOBASHI MAPS!!!!
        kw,kj,kh,jw,hw,Hap =  find_wband_cols(-model[i]['RAv'],model[i],typ)
        q_cols.append(round(jw + (1.85*hw),3))

        H_mags.append(round(Hap,3))
        col1.append(round(jw,3))	
        col2.append(round(hw,3))

    H_apcol = Column(H_mags,name='H_app')
    Jw_col = Column(col1,name='J-W')
    Hw_col = Column(col2,name='H-W')
    q_col_col = Column(q_cols,name='Q(cols)')

    model.add_columns([H_apcol,Jw_col,Hw_col,q_col_col])
    return model


# IMF FUNCTIONS: FIT TO POPULATION BY ADJUSTING THE VARIABLES
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def kroupa_imf(m,a):
#The values given for these in their paper are 0.3 and 1.3
    if m < 0.08:
        e = m**-a[0]
    elif m >= 0.08 and m <=0.5:
        e = m**-a[1]
    else:
        e = m**-2.3
    return e

def chab_log_imf(m,a):
#According to trilegal paper, they use m0 = 0.1, sig = 0.627 - but dont specifiy what the multiplying constant is
#Chabrier03 paper has A=0.158, m0 = 0.079, sig = 0.69
#Not sure what form to use: second line comes straight from Chab03 paper
    if m < 1.0:
        e = a[0]*(1./(np.log(10.)*m))*np.exp(-(np.log10(m)-np.log10(a[1]))**2./(2*a[2]**2))
        #e = a[0]*np.exp(-(np.log10(m)-np.log10(a[1]))**2./(2*a[2]**2))  
    else:
        e = (4.43E-2)*m**-1.3
    return e

def chab_exp_imf(m,A,B,m0):
    m = A*m**-a*(np.exp(-(m0/m)**b))
    return m


def salp_imf(m,a):
    e = m**a[0]
    return e

def create_salp_background():
    
    #Full sample of objects should be same length as background-corrected observed: 11719
    nobjs = 1000
    e=[]

    mlisttab = Table.read('teff_spt_table.txt',format='ascii')
    fullm = np.asarray(mlisttab['Msun'][6:97])
    for i in range(len(fullm)):
        e.append(salp_imf(fullm[i],[-2.35]))

    e = [a/max(e) for a in e]
    backg_obj_masses = choices(fullm,weights=e,k=nobjs)
    mass_col = Column(backg_obj_masses,name='Mass(msol)')

    alist = np.arange(0.25,30.25,0.25)
    av_list = np.random.choice(alist,nobjs)
    #av_list = np.random.normal(10.0,2.0,nobjs)
    av_col = Column(av_list,name='RAv')

    dlist = np.arange(10.,400.,0.5)
    d = np.random.choice(dlist,nobjs)
    d_col = Column(d,name='Dist(pc)')

    t = Table()
    t.add_columns([mass_col,av_col,d_col])
    t = add_spt(t,'Msun','Mass(msol)','b')
    return t

# FUNCTION TO CREATE CREATE POPULATION OF YOUNG OBJECTS IN SERPENS CLUSTER - THIS NEEDS TO BE OPTIMISED TO BEST REPRODUCE OBSERVED QVSH WITH BACKGROUND REMOVED
# INPUTS: - IMF INFORMATION i.e. distribution of masses to pick from
#		  - Av INFORMATION (same one as background pop in main)
#		  - DEPTH OF CLUSTER (Herczeg+19, d = 430+/-30)
# - - - - - - - - - - - -  - - - -- - - - - - - - - - - - - - - - - - - - - - 
def create_young_pop(imf,a,difflen):
    #Remove last element of parameter of array as this is not an IMF parameter
   # a = a[0:len(a)-1]
    #Possible imf keywords: Chabrier-log ,Kroupa
    nsamp = 250
    #Full sample of objects should be same length as background-corrected observed: 11719
    nobjs = difflen
    t,e = [],[]
    full_m = np.linspace(0.07,0.4,nsamp)

    start = time.time()
    
    if 'chab' in imf:
        for i in range(nsamp):
    		#Generate FULL distribution of masses and probabilities, to a fine resolution: e = number of each mass, need to normalise?
            e.append(chab_log_imf(full_m[i],a))

    elif 'kroup' in imf:
        for i in range(nsamp):
            e.append(kroupa_imf(full_m[i],a))

    elif 'salp':
        for i in range(nsamp):
            e.append(salp_imf(full_m[i],a))

        #Masses of objects   
    e = [a/max(e) for a in e]

    young_obj_masses = choices(full_m,weights=e,k=nobjs)
    end = time.time()
    #print('Took',str(end-start))
    mass_col = Column(young_obj_masses,name='Mass(msol)')
    
    #Extinctions of objects: random choices from a distribution with weights
    #av_dist = Table.read('serpens_extinction_probabilities/ss_av_converted_prob_density_higher_conversion.txt',format='ascii')
    #av_list = choices(av_dist['Av Value'],weights=av_dist['Frequency'],k=nobjs)

    #Extinctions: random choices without weights
    #alist = np.arange(10.0,30.25,0.25)
    alist = np.arange(0.0,5.0,0.25)
    av_list = np.random.choice(alist,nobjs)
    #av_list = [0 for i in range(nobjs)]

    av_col = Column(av_list,name='RAv')
    
    t = Table()    
    #Option to have distance variation
    DIST = False

    if DIST:
    #Distances of objects - just use gaussian distribution for now
        dist = [438,11]
        dist_list = []
        for i in range(nsamp*100):
            dist_list.append(np.random.normal(dist[0],dist[1]))

        dhist,dvals = np.histogram(dist_list,bins=10000)
        dvals = dvals[0:len(dvals)-1]
        d_list = choices(dvals,weights=dhist,k=nobjs)
        d_col = Column(d_list,name='Dist(pc)')
        t.add_columns([mass_col,av_col,d_col])
    t.add_columns([mass_col,av_col])

    start = time.time()
    t = add_spt(t,'Msun','Mass(msol)','y')
    end = time.time()
    #print('Took',str(end-start))

    return t

# FUNCTION TO CALL MULTIPROCESSING LOOP FOR GENERATING BACKGROUND POPULATION
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def background_popQH():

    #check that this is correct model!
    model = Table.read('trilegal_files/taurus_trilegal_8835_KNB.txt',format='ascii')
    #bs_list = bs_list [0:400]
    RANDAV = True
    NOSPT = True

    # CREATE BACKGROUND POPULATION USING TRILEGAL MODEL TABLE
    # Option to give each object a randomly chosen Av from a probability distribution
    if RANDAV:
        #av_dist = Table.read('serpens_extinction_probabilities/ss_av_converted_prob_density.txt',format='ascii')
        #av_list = choices(av_dist['Av Value'],weights=av_dist['Frequency'],k=len(model))
        alist = np.arange(0.0,5.5,0.25)
        av_list = np.random.choice(alist,len(model))
        av_col = Column(av_list,name='RAv')
        model.add_column(av_col)

    if NOSPT:
        model = add_spt(model,'logT','logTe','t')	

    chunks = [model[i::16] for i in range(16)]
    pool = Pool(processes=16)
    result = pool.map(mp_loop_single_wband_table,chunks)

    #mp_loop_single_wband_table(model)

    full1 = join(result[0],result[1],join_type='outer')
    full2 = join(result[2],result[3],join_type='outer')
    full3 = join(result[4],result[5],join_type='outer')
    full4 = join(result[6],result[7],join_type='outer')
    full5 = join(result[8],result[9],join_type='outer')
    full6 = join(result[10],result[11],join_type='outer')
    full7 = join(result[12],result[13],join_type='outer')
    full8 = join(result[14],result[15],join_type='outer')

    quar1 = join(full1,full2,join_type='outer')
    quar2 = join(full3,full4,join_type='outer')
    quar3 = join(full5,full6,join_type='outer')
    quar4 = join(full7,full8,join_type='outer')

    half1 = join(quar1,quar2,join_type='outer')
    half2 = join(quar3,quar4,join_type='outer')

    background = join(half1,half2,join_type='outer')

    return background


# FUNCTION TO CALL MULTIPROCESSING LOOP FOR GENERATING YOUNG SERPENS POPULATION
# - - - - - - - - - - - - -- - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - -- 
def create_popQH(tab):

    chunks = [tab[i::4] for i in range(4)]
    pool = Pool(processes=4)
    result = pool.map(mp_loop_single_wband_table,chunks)

    full1 = join(result[0],result[1],join_type='outer')
    full2 = join(result[2],result[3],join_type='outer')
    table = join(full1,full2,join_type='outer')

    #young.write('population_files/young_chabIMF_dobashAv_oneD.txt',format='ascii',overwrite=True)
    
    return table


   # FUNCTION TO CREATE REFERENCE GRID OF SPT VS AV 

def make_grid():

    tab = Table.read('l17_files_qplot.txt',format='ascii')
    stds = tab['Typ']
    #av_dist = Table.read('serpens_extinction_probabilities/ss_av_converted_prob_density_higher_conversion.txt',format='ascii')

    #av_min = min(av_dist['Av Value'])
    #av_max = max(av_dist['Av Value'])
    av_min = 0.0
    av_max = 30.25
    av_list = np.arange(av_min,av_max+0.25,0.25)

    model = Table()
    typ,av,q_cols,H_mags = [],[],[],[]

    for i in range(len(stds)):
        print(i) 
        for j in range(len(av_list)):   
            kw,kj,kh,jw,hw,Hap =  find_wband_cols(-av_list[j],stds[i],'realfile')
            q_cols.append(round(jw + (1.85*hw),3))
            H_mags.append(round(Hap,3))
            typ.append(stds[i])
            av.append(av_list[j])
    
    typ_col = Column(typ,name='SpT')
    av_col = Column(av,name='RAv')
    H_apcol = Column(H_mags,name='H_app')
    q_col_col = Column(q_cols,name='Q(cols)')

    model.add_columns([typ_col,av_col,H_apcol,q_col_col])
    model.write('spt_av_grid_0-30_converted_av_higher.txt',format='ascii')

    return model 


def young_lookup(tab):
    
    lookup = Table.read('spt_av_grid_0-30_converted_av_higher.txt',format='ascii')
    rem_ind = [] 
    index_list = []
    
    av_dist = Table.read('serpens_extinction_probabilities/ss_av_converted_prob_density_higher_conversion.txt',format='ascii')
    #av_min = min(av_dist['Av Value'])
    #av_max = max(av_dist['Av Value'])
    av_min = 0.0
    av_max = 30.25
    av_list = np.arange(av_min,av_max+0.25,0.25)

    for i in range(len(tab)):
   #     print(tab[i])
        c = np.nan
        
        yav = tab[i]['RAv']
        sptind = tab[i]['SpTind']
        #print(sptind,yav) 
        for k in range(len(av_list)):    
            if yav == av_list[k]:
                av_ind = k
    #            print(av_ind)
       	total_ind = (sptind*len(av_list))+av_ind
    #    print(total_ind)
        index_list.append(total_ind)
        #if np.isnan(c):
        #    rem_ind.append(i)
        #    n=n+1

    hcol = lookup['H_app'][index_list]
    qcol = lookup['Q(cols)'][index_list]

    tab.add_columns([hcol,qcol])
    tab.remove_rows([rem_ind])

    return tab

def salpeter_pop(a):
    # N is a parameter of emcee, and is used to dictate length of random sample chosen from full Salpeter population
    salp = Table.read('population_files/broad_salpeter_3000.txt',format='ascii') 
    n = int(a[2])
    #choose random n rows from full table
    rows = np.arange(0,len(salp),1)
    rand_rows = np.random.choice(rows,n)
    new_salp = salp[rand_rows]

    return new_salp

# FUNCTION TO FIND LEFT OVER POPULATION OF QvsH PLOT WITH BACKGROUND REMOVED, AND TO PLOT GENERATED YOUNG POPULATION
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def define_populations(PLOTS,salp):

    rem_o,rem_t,rem_s,rem_y=[],[],[],[]
    qlimu = -0.6
    qliml = -2.5
    comb = True

    # DEFINE BINS OVER PARAMETER RANGE OF INTEREST
    xbins = np.arange(14,21,0.2)
    ybins = np.arange(qliml,qlimu,0.2)
    
    # REMOVE NANS AND OTHER PARAMETER VALUES FROM SALPETER TABLE
    #salp=Table.read('population_files/broad_salpeter_1000.txt',format='ascii')
    for i in range(len(salp)):
        if np.isnan(salp[i]['H_app']) or np.isnan(salp[i]['Q(cols)']):
            rem_s.append(i)
    salp.remove_rows([rem_s])
    full_salp_len = len(salp)
    rem_s = []
    for i in range(len(salp)):
        if salp['H_app'][i] < 14 or salp['H_app'][i] > 21 or salp['Q(cols)'][i] < qliml or salp['Q(cols)'][i] > qlimu:
            rem_s.append(i)
    salp.remove_rows([rem_s])
    slen = len(salp)

    # REMOVE NANS AND OTHER PARAMETER VALUES FROM TRILEGAL TABLE
    #tril = Table.read('population_files/backg_kroupBIMF_dobashiAj_Ls_Ts.txt',format='ascii')
    tril = Table.read('population_files/taurus_backg_kroupNBIMF_0-5Av.txt',format='ascii')
    for i in range(len(tril)):
        if np.isnan(tril[i]['H_app']):
            rem_t.append(i)
    tril.remove_rows([rem_t])
    full_tril_len = len(tril)
    rem_t = []
    for i in range(len(tril)):
       if tril['H_app'][i] < 14 or tril['H_app'][i] > 21 or tril['Q(cols)'][i] < qliml or tril['Q(cols)'][i] > qlimu: 
           rem_t.append(i)
    tril.remove_rows([rem_t])
    tlen = len(tril)

    if slen == 0:
        backg = tril
        blen = tlen
        full_back_len = full_tril_len
    else:
        backg = join(salp,tril,join_type='outer')
        blen = tlen+slen
        full_back_len = full_salp_len + full_tril_len
    # MAKE HISTOGRAM OF OBJECTS FROM BACKGROUND POP AND NORMALISE   
    oBQH,x,y = np.histogram2d(backg['H_app'],backg['Q(cols)'],[xbins,ybins],range=[[14,21],[qliml,qlimu]])
    BQH = np.asarray([a/blen for a in oBQH])

    if PLOTS:
        plt.figure()
        plt.imshow(BQH.T, interpolation='nearest',origin='lower',extent=[14,21,qliml,qlimu])
        plt.xlim(14,21)
        plt.ylim(qliml,qlimu)
        plt.title('Background pop.')
        plt.colorbar()
        plt.show()
        #plt.close()
    
    # REMOVE OTHER PARAMETER VALUES FROM OBSERVED POPULATION 
    # SERPENS OBSERVED POP:
    #QHtab = Table.read('QvsH_data_notnan_ss.txt',format='ascii')

    # SYNTHETIC 'OBSERVED" POP:
    #QHtab = Table.read('population_files/young_salpeter_serpAv_ext.txt',format='ascii')

    # TAURUS 8335 OBSERVED POP
    QHtab = Table.read('taurus/t_7am_QvsH.txt',format='ascii')

    full_obs_len = len(QHtab)
    for i in range(len(QHtab)):
        if QHtab['H'][i] <14 or QHtab['H'][i]>21 or QHtab['Q'][i] >qlimu or QHtab['Q'][i] <qliml or np.isnan(QHtab['H'][i]):
            rem_o.append(i)
    QHtab.remove_rows([rem_o])
    olen =len(QHtab)
    
    # MAKE HISTOGRAM OF OBJECTS FROM OBSERVED POPULATION AND NORMALISE
    oQH,x,y = np.histogram2d(QHtab['H'],QHtab['Q'],[xbins,ybins],[[14, 21], [qliml,qlimu]])
    QH = np.asarray([a/olen for a in oQH])

    if PLOTS:
        plt.figure()
        plt.imshow(QH.T, interpolation='nearest',origin='lower',extent=[14,21,qliml,qlimu])
        plt.xlim(14,21)
        plt.ylim(qliml,qlimu)
        plt.title('Observed pop.')
        plt.colorbar()
        plt.show()
        #plt.close()
  
    # THIS SECTION IS UNIMPORTANT, GOING TO BE GENERATING YOUNG RATHER THAN READING IN
    young = Table.read('population_files/synthetic_kroupa_taurus_av.txt',format='ascii')
    for i in range(len(young)):
        if young['H_app'][i] < 14 or young['H_app'][i] > 21 or young['Q(cols)'][i] < qliml or young['Q(cols)'][i] > qlimu: 
            rem_y.append(i)
    young.remove_rows([rem_y])
    ylen = len(young)

    YQH,x,y = np.histogram2d(young['H_app'],young['Q(cols)'],[xbins,ybins],range=[[14,21],[qliml,qlimu]])
    YQH = np.asarray([a/ylen for a in YQH])

    if PLOTS:
        plt.figure()
        plt.imshow(YQH.T, interpolation='nearest',origin='lower',extent=[14,21,qliml,qlimu])
        plt.xlim(14,21)
        plt.ylim(qliml,qlimu)
        plt.title('Young pop.')
        plt.ylabel('Q')
        plt.xlabel('H mag')
        plt.colorbar()
        plt.show()

    # CREATE ERROR HISTOGRAMS FOR EACH OF THE DISTRIBUTIONS - square root the bin values and divide by table len
    eBQH = np.asarray([np.sqrt(a)/blen for a in oBQH])
    difflen = full_obs_len-full_back_len    

    return QH,BQH,eBQH,difflen


def create_young_hist(young,PLOTS):

    qlimu = -0.6
    qliml = -2.5
    xbins = np.arange(14,21,0.2)
    ybins = np.arange(qliml,qlimu,0.2)

    rem_y = []
    for i in range(len(young)):
        if young['H_app'][i] < 14 or young['H_app'][i] > 21 or young['Q(cols)'][i] < qliml or young['Q(cols)'][i] > qlimu:
            rem_y.append(i)
    young.remove_rows([rem_y])
    ylen = len(young)

    oYQH,x,y = np.histogram2d(young['H_app'],young['Q(cols)'],[xbins,ybins],range=[[14,21],[-3.2,-0.6]])
    YQH = np.asarray([a/ylen for a in oYQH])
    eYQH = np.asarray([np.sqrt(a)/ylen for a in oYQH])

    if PLOTS:
        plt.figure()
        plt.imshow(YQH.T, interpolation='nearest',origin='lower',extent=[14,21,qliml,qlimu])
        plt.xlim(14,21)
        plt.ylim(qliml,qlimu)
        plt.title('Young pop.')
        plt.ylabel('Q')
        plt.xlabel('H mag')
        plt.colorbar()
        plt.show()

    return YQH,eYQH

def chisqr(d,y,ye):
        if ye == 0:
                return 0.0
        else:
                inve2 = 1/((ye)**2)
                return (d-(y))**2*inve2

def chisq_test():

	#a1s = [0.0,0.3,0.6,0.6]
	#a2s = [0.0,1.3,2.6,1.05]

	a1s = np.linspace(0.0,1.0,11)
	a2s = np.linspace(0.0,2.5,11)
	result = [[0 for i in range(len(a1s))] for j in range(len(a2s))]

	for k in range(len(a1s)):
		for l in range(len(a2s)):
			print(a1s[k],a2s[l])
			salptab = Table.read('population_files/broad_salpeter_3000.txt',format='ascii')
			ohist,bhist,bhist_error,difflen = define_populations(False,salptab)
			tab = create_young_pop('kroupa',[a1s[k],a2s[l]],10000)
			young = young_lookup(tab)
			yhist,yhist_error = create_young_hist(young,False)
			chi = []
			for i in range(len(yhist)):
				for j in range(len(yhist[0])):
                       #print('Obs = ',str(obshist[i][j]),' Young = ',str(yhist[i][j]))
					chi.append(chisqr(ohist[i][j],yhist[i][j],yhist_error[i][j]))
			likeli = np.sum(np.asarray(chi))
			redchi = likeli/(len(chi)-2)
			lk = np.exp(-redchi/2.)
			print(lk)
			result[k][l] = lk
	
	fig,ax = plt.subplots(figsize=(10,10))
	c=ax.imshow(res,extent=[min(a1s),max(a1s),min(a2s),max(a2s)])
	ax.set_aspect(0.5)
	plt.colorbar(c)
	plt.scatter(0.6,1.05,c='red')
	plt.show()
		
	return result


#histogram()
b = background_popQH()
#y,tab = create_young_pop('kroupa')
#young = mp_loop_single_wband_table(tab)
#young = young_popQH(tab)
#tab = find_leftover_pop()

#dif = define_populations(True)
b.write('population_files/taurus_backg_kroupNBIMF_0-5Av_mult_stds.txt',format='ascii')
#make_grid()
