import pandas
import numpy
import datetime
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import hdbscan
import shapely.geometry as geometry
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
from matplotlib import ticker
from matplotlib.dates import date2num, DateFormatter, MinuteLocator
from matplotlib.ticker import FuncFormatter, MaxNLocator
from davitpy import utils

class PCPCluster(object):
    """
    A class to read in data from model files and generate
    a timeseries plot of average TEC value binned by latitude,
    to test the accuracy of the model.
    """
    def __init__(self, dateRange, mfTECDir="/sd-data/med_filt_tec/",\
             make_plot=True, plotDir="/home/bharat/Documents/pcp-plots/",\
             pltTypes=["pcp-cluster-bnd", "clusters"], eps=0.07,\
             clusterType="HDBSCAN", minSamples=5, metric="precomputed"):
        # get a list of TEC files to be read from the dateRange
        # remember we could have multiple days! so we'll have to
        # read multiple days if necessary!
        dfList = []
        delDays = (dateRange[1].date() - dateRange[0].date()).days
        inpColList = [ "dateStr", "timeStr", "Mlat",\
               "Mlon", "med_tec", "dlat", "dlon" ]
        # get the days and read the data
        if delDays > 0:
            print "reading data!"
            for _d in range(delDays+1):
                _cd = dateRange[0] + datetime.timedelta(days=_d)
                _tecfl = mfTECDir + "tec-medFilt-" +\
                                 _cd.strftime("%Y%m%d") + ".txt"
                dfList.append( pandas.read_csv(_tecfl,\
                                 delim_whitespace=True,
                                header=None, names=inpColList) )
        self.tecDF = pandas.concat(dfList)
        self.tecDF["date"] = self.tecDF.apply( self.convert_to_datetime,\
                                 axis=1 )
        # Limit the DF to the input time Range and to a few cols
        self.tecDF = self.tecDF[ (self.tecDF["date"] >= dateRange[0]) &\
                                (self.tecDF["date"] <= dateRange[1])\
                                 ][ [ "Mlat", "Mlon", "med_tec", "date"\
                                  ] ].reset_index(drop=True)
        # clustering settings
        self.clusterType = clusterType
        self.eps = eps
        self.minSamples = minSamples
        self.metric = metric
        # plotting settings
        self.make_plot = make_plot
        self.pltTypes = pltTypes
        self.plotDir = plotDir
        
    def convert_to_datetime(self, row):
        currDateStr = str( int( row["dateStr"] ) )
    #     return currDateStr
        if row["timeStr"] < 10:
            currTimeStr = "000" + str( int( row["timeStr"] ) )
        elif row["timeStr"] < 100:
            currTimeStr = "00" + str( int( row["timeStr"] ) )
        elif row["timeStr"] < 1000:
            currTimeStr = "0" + str( int( row["timeStr"] ) )
        else:
            currTimeStr = str( int( row["timeStr"] ) )
        return datetime.datetime.strptime( currDateStr\
                        + ":" + currTimeStr, "%Y%m%d:%H%M" )

    def cluster_data(self):
        # for each time interval cluster the data
        uniqDates = self.tecDF["date"].unique()
        # loop through each date and get the clusters
        for _cd in uniqDates:
            print "currently processing---->", _cd
            selTecDF = self.tecDF[ self.tecDF["date"] == _cd\
                         ].reset_index(drop=True)
            # Re-scale Mlon to -180 to 180 (from 0 to 360)
            selTecDF["adjstMlons"] = [ b - 360. if b > 180. else\
                         b for b in selTecDF["Mlon"] ]
            # limit analysis to lats > 50.
            selTecDF = selTecDF[ selTecDF["Mlat"] >= 50. ]
            # we'll round the tec values for improved clustering
            selTecDF["rnd_tec"] = selTecDF["med_tec"].apply(\
                                lambda x: self.custom_round(x, base=10))
            # get the weight matrix for distance calculation
            if self.metric == "precomputed":
                print "calculating weight matrix"
                wghtMatrix = self.weight_matrix_pcp(selTecDF)
            else:
                wghtMatrix = selTecDF[ [ "Mlat", "adjstMlons", "med_tec" ]\
                                 ].as_matrix()
            # Now get to the actual DBSCAN
            if self.clusterType == "DBSCAN":
                dbsc = DBSCAN(eps=self.eps, min_samples=self.minSamples,\
                             metric=self.metric).fit(wghtMatrix)
            else:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=self.minSamples,\
                            metric=self.metric)
                dbsc = clusterer.fit(wghtMatrix)
            # Number of clusters in labels, ignoring noise if present.
            nClusters = len(set(dbsc.labels_)) - (1 if -1 in dbsc.labels_ else 0)
            print "Estimated number of clusters---->", nClusters
            # Now generate a plot
            if self.make_plot:
                self.generate_plot(selTecDF, dbsc)

    def generate_plot(self, selTecDF, dbsc, pltCoords="mag"):
        # generate plots to show the clusters detected!
        # set styling
        sns.set_style("whitegrid")
        # get the patch data
        patchTS = (selTecDF["date"].unique()[0] - numpy.datetime64(\
                '1970-01-01T00:00:00Z')) / numpy.timedelta64(1, 's')
        patchDate = datetime.datetime.utcfromtimestamp(patchTS)
        labels = dbsc.labels_
        uniqLabels = set(labels)
        print "plotting clusters"
        # First plot type
        if "pcp-cluster-bnd" in self.pltTypes:
            tecDataArr = selTecDF[ [ "Mlat", "adjstMlons", "med_tec" ]\
                                 ].as_matrix()
            # set colorbar
            seaMap = "jet"#ListedColormap(sns.color_palette("Spectral_r"))
            f = plt.figure(figsize=(12, 8))
            ax1 = f.add_subplot(1,1,1)
            m1 = utils.plotUtils.mapObj(boundinglat=50,\
                        coords=pltCoords, ax=ax1, datetime=patchDate)
            # Plot the actual data
            xVec, yVec = m1(list(selTecDF["Mlon"]),\
                            list(selTecDF["Mlat"]), coords=pltCoords)
            tecPlot = m1.scatter( xVec, yVec , c=selTecDF["med_tec"], s=40.,\
                       cmap=seaMap, alpha=0.7, zorder=5., \
                                 edgecolor='none', marker="s",\
                                vmin=0, vmax=25)
            cbar = plt.colorbar(tecPlot, orientation='vertical',\
                     ax=ax1, shrink=0.9)
            cbar.set_label('TEC', size=15)
            # Plot the PCP boundaries
            tecDataArr = selTecDF[ [ "Mlat", "adjstMlons", "med_tec"\
                             ] ].as_matrix()
            for _n, _k in enumerate(uniqLabels):
                # plot the boundaries of each cluster
                clsMbrMask = (labels == _k)    
                xy = tecDataArr[clsMbrMask]
                # Try and find only pcp clusters
                if numpy.mean(xy[:,2]) < 20:
                    continue
                if xy.shape[0] > 100:
                    continue
                if numpy.mean(xy[:,0]) < 60:
                    continue
                # Get boundary using shapely
                shpPnts = [ geometry.Point( _x, _y ) for _x, _y\
                               in zip(xy[:,0].ravel(), xy[:,1].ravel()) ]
                shpPnts = geometry.MultiPoint(list(shpPnts))
                bndCrdDict = shpPnts.convex_hull.boundary.__geo_interface__
                bndLatList = []
                bndLonList = []
                for _b in bndCrdDict["coordinates"]:
                    bndLatList.append( _b[0] )
                    bndLonList.append( _b[1] )
                m1.plot(bndLonList, bndLatList,'k-', linewidth=3, zorder=7., latlon=True)
            pltFName = self.plotDir + self.clusterType + "-pcp-bnd-" +\
                         patchDate.strftime("%Y%m%d-%H%M") + ".pdf"
            ax1.set_title(patchDate.strftime("%Y%m%d-%H%M"))
            f.savefig(pltFName,bbox_inches='tight')
        # Another plot type
        if "clusters" in self.pltTypes:
            # TEC data arr
            tecDataArr = selTecDF[ [ "Mlat", "adjstMlons", "med_tec" ]\
                            ].as_matrix()
            # set colorbar
            seaMap = ListedColormap(sns.color_palette("Spectral_r"))
            f = plt.figure(figsize=(12, 8))
            ax1 = f.add_subplot(1,2,1)
            ax2 = f.add_subplot(1,2,2)
            # Original map
            m1 = utils.plotUtils.mapObj(boundinglat=50,\
                        coords=pltCoords, ax=ax1, datetime=patchDate)

            xVec, yVec = m1(list(selTecDF["Mlon"]),\
                            list(selTecDF["Mlat"]), coords=pltCoords)
            tecPlot = m1.scatter( xVec, yVec , c=selTecDF["med_tec"], s=40.,\
                       cmap=seaMap, alpha=0.7, zorder=5., \
                                 edgecolor='none', marker="s" )
            cbar = plt.colorbar(tecPlot, orientation='horizontal',\
                         ax=ax1, shrink=0.9, pad=0.0)
            # Clustered map
            m2 = utils.plotUtils.mapObj(boundinglat=50,\
                        coords=pltCoords, ax=ax2, datetime=patchDate)
            # colors = [plt.cm.Dark2(each)
            #           for each in np.linspace(0, 1, len(uniqLabels))]
            colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
            for _n, _k in enumerate(uniqLabels):
                if _k == -1:
                    # Black used for noise.
                    col = 'k'
                    continue
                else:
                    col = colors[ colors.keys()[_n] ]
                clsMbrMask = (labels == _k)
                if self.clusterType == "DBSCAN":     
                    coreSmplMask = numpy.zeros_like(labels, dtype=bool)
                    coreSmplMask[dbsc.core_sample_indices_] = True    
                    xy = tecDataArr[clsMbrMask & coreSmplMask]
                    clsPlt = m2.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                             markeredgecolor='k', markersize=6, latlon=True)

                    xy = tecDataArr[clsMbrMask & ~coreSmplMask]
                    m2.plot(xy[:, 1], xy[:, 0], 'x', markerfacecolor=col,
                             markeredgecolor='k', markersize=6, latlon=True)
                else:
                    xy = tecDataArr[clsMbrMask]
                    clsPlt = m2.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                             markeredgecolor='k', markersize=6, latlon=True)
                    m2.plot(xy[:, 1], xy[:, 0], 'x', markerfacecolor=col,
                             markeredgecolor='k', markersize=6, latlon=True)
            pltFName = self.plotDir + self.clusterType + "-clusters-" +\
                     patchDate.strftime("%Y%m%d-%H%M") + ".pdf"
            plt.title(patchDate.strftime("%Y%m%d-%H%M"))
            f.savefig(pltFName,bbox_inches='tight')

    def custom_round(self, x, base=10):
        return int(base * round(float(x)/base))

    def weight_matrix_pcp(self,tecDF):
        # We are not using the default Mlon/Mlat weighting method
        # we are using a custom method where we calculate the distance
        # using a custom func! This involves using the haversine formula
        # to estimate the great circle distance and then adding a TEC
        # closeness measure!
        # create a distance matrix
        # matrix dimensions
        latList = tecDF["Mlat"].values
        lonList = tecDF["Mlon"].values
        tecList = tecDF["rnd_tec"].values
        nRecs = tecDF.shape[0]
        wghtMatrix = numpy.zeros((nRecs, nRecs))
        aa = []
        bb = []
        for i in range(nRecs):
            for j in range(nRecs):
                if wghtMatrix[i, j] == 0.0:
                    
                    # convert decimal degrees to radians 
                    lon1, lat1, lon2, lat2 = map( radians, [lonList[i], latList[i],\
                                                           lonList[j], latList[j]] )
                    # haversine formula 
                    dlon = lon2 - lon1 
                    dlat = lat2 - lat1 
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    
                    tecDiff = sqrt(numpy.abs( tecList[i] - tecList[j] ))/4.
                    wghtMatrix[i, j] = c + tecDiff
                    wghtMatrix[j, i] = wghtMatrix[i,j]
        return wghtMatrix

if __name__ == "__main__":
    dateRange = [ datetime.datetime(2013,1,17,20, 25),\
                     datetime.datetime(2013,1,18,3, 30) ]
    pcpObj = PCPCluster(dateRange)
    pcpObj.cluster_data()