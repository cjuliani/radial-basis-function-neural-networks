# Radial basis function neural network for mineral exploration risk calculation

The demo was initially developed for prospectivity mapping of copper-rich mineral deposits within Northern Norway. Geoscientific spatial data (lithology, magnetics, geochemistry and gravity) are combined and matched with existing mineral deposits located in the area. The goal was to find geoscientific "fingerprints" from this combining matching these deposits, and to seek similar fingerprints elsewhere in the survey area. 

Original paper:
Juliani, C., Ellefmo, S.L., 2019. Prospectivity Mapping of Mineral Deposits in Northern Norway Using Radial Basis Function Neural Networks. Minerals, 9(2). DOI:10.3390/min9020131

**NOTES**

    Data examples correspond to a CSV file with:
    - oid: ArcGIS grid identification number associated to a coordinate ({x,y}, UTM)
    - geology,gravity,magnetics,U,K,Th: normalized geoscientific data spatially extracted from a grid
    - deposits: labelled 0 to 5 | 0: no deposit; 4: copper-rich deposit; {1,2,3,5}: other deposit types

*Make sure you use Python 3.x and that related libraries (numpy and tensorflow) have correct versions for compatibility.*

![alt text](https://www.mdpi.com/minerals/minerals-09-00131/article_deploy/html/images/minerals-09-00131-g008-550.jpg)
