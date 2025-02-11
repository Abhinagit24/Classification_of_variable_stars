# importing the libraries
from astropy.io import ascii

# converting the mrt format file to a pandas df
print("Enter the name of the MRT file:")
file_name = input()
table = ascii.read(file_name, format='mrt')
df = table.to_pandas()
df.to_csv('drgreg_dscuti.csv')

