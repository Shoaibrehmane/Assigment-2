
#import important libraries
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 



# ### Implement a Function Which return Original DataFrame, Transposed DataFrames

def data_transpose(filename: str):
    
    # Read the file into a pandas dataframe
    dataframe = pd.read_csv(filename)
    
    # Transpose the dataframe
    df_transposed = dataframe.transpose()
    
    # Populate the header of the transposed dataframe with the header information 
   
    # silice the dataframe to get the year as columns
    df_transposed.columns = df_transposed.iloc[1]
    # As year is now columns so we don't need it as rows
    df_transposed_year = df_transposed[0:].drop('year')
    
    # silice the dataframe to get the country as columns
    df_transposed.columns = df_transposed.iloc[0]
    
    # As country is now columns so we don't need it as rows
    df_transposed_country = df_transposed[0:].drop('country')
    
    return dataframe, df_transposed_country, df_transposed_year



# Passing filename to worldbank_data_transpose function 
orig_df, country_as_col, year_as_col = data_transpose('climatechange_dataset.csv')


# ### Original DataFrame

# show the first 5 row
orig_df.head(5)


# ### DataFrame In Which Countries are Columns


# show the first 5 row
# country as col
country_as_col.head(5)


# ### show the statistics of Original Data

#describe method show the statistic of dataframe
orig_df.describe() 


# ### DataFrame In Which Year are Columns

# show the first 5 row
# year as col
year_as_col.head(5)


# countries fresh water data over specfic years
# we need to extract data from our original data frame
fresh_water_data = orig_df[['country','year','fresh_water']]

# drop the null values present in the dataset
fresh_water_data = fresh_water_data.dropna()


# ### Get Data to Specific Years from 1990 to 2020

# data related to 1990 
no_data_1990 = fresh_water_data[fresh_water_data['year'] == 1990] 

# data related to 1995
no_data_1995 = fresh_water_data[fresh_water_data['year'] == 1995] 

# data related to 2000
no_data_2000 = fresh_water_data[fresh_water_data['year'] == 2000] 

# data related to 2005 
no_data_2005 = fresh_water_data[fresh_water_data['year'] == 2005] 

# data related to 2010 
no_data_2010 = fresh_water_data[fresh_water_data['year'] == 2010]

# data related to 2015 
no_data_2015 = fresh_water_data[fresh_water_data['year'] == 2015]

# data related to 2020 
no_data_2020 = fresh_water_data[fresh_water_data['year'] == 2020] 


style.use('ggplot')

# set fig size
plt.figure(figsize=(15,10))

# set width of bars
barWidth = 0.1

# plot bar charts
plt.bar(np.arange(no_data_1990.shape[0]),
        no_data_1990['fresh_water'],
        color='goldenrod', width=barWidth, label='1990')

plt.bar(np.arange(no_data_1995.shape[0])+0.2,
        no_data_1995['fresh_water'],
        color='blue',width=barWidth, label='1995')

plt.bar(np.arange(no_data_2000.shape[0])+0.3,
        no_data_2000['fresh_water'],
        color='greenyellow',width=barWidth, label='2000')

plt.bar(np.arange(no_data_2005.shape[0])+0.4,
        no_data_2005['fresh_water'],
        color='olive',width=barWidth, label='2005')

plt.bar(np.arange(no_data_2010.shape[0])+0.5,
        no_data_2010['fresh_water'],
        color='dodgerblue',width=barWidth, label='2010')

plt.bar(np.arange(no_data_2015.shape[0])+0.6,
        no_data_2015['fresh_water'],
        color='slategray',width=barWidth, label='2015')




# show the legends on the plot
plt.legend()

# set the x-axis label
plt.xlabel('Country',fontsize=15)

# add title to the plot 
plt.title("Fresh Water",fontsize=15)

# add countries names to the 11 groups on the x-axis
plt.xticks(np.arange(no_data_2010.shape[0])+0.2,
           ('United Arab Emirates', 'Armenia', 'Belgium', 'France',
       'Indonesia', 'Kenya', 'Nepal', 'Saudi Arabia', 'Sweden',
       'Eswatini', 'Tajikistan'),
           fontsize=10,rotation = 45)

# show the plot
plt.show()



# we want to see countries urban_population over the years
urban_population = orig_df[['country','year','urban_population']]

# drop the null values present in the dataset
urban_population  = urban_population.dropna()



# ### Filter from specific year from 1990 to 2015

# data related to 1990
data_1990 = urban_population[urban_population['year'] == 1990]

# data related to 1995
data_1995 = urban_population[urban_population['year'] == 1995]

# data related to 2000
data_2000 = urban_population[urban_population['year'] == 2000]

# data related to 2005
data_2005 = urban_population[urban_population['year'] == 2005] 

# data related to 2010
data_2010 = urban_population[urban_population['year'] == 2010]

# data related to 2015 
data_2015 = urban_population[urban_population['year'] == 2015] 

# data related to 2020
data_2020 = urban_population[urban_population['year'] == 2020]


# ### PLOT barplot

style.use('ggplot')

# set fig size
plt.figure(figsize=(15,10))

# set width of bars
barWidth = 0.1 

# plot bar charts
plt.bar(np.arange(data_1990.shape[0]),
        data_1990['urban_population'],
        color='goldenrod', width=barWidth, label='1990')

plt.bar(np.arange(data_1995.shape[0])+0.2,
        data_1995['urban_population'],
        color='blue',width=barWidth, label='1995')

plt.bar(np.arange(data_2000.shape[0])+0.3,
        data_2000['urban_population'],
        color='greenyellow',width=barWidth, label='2000')

plt.bar(np.arange(data_2005.shape[0])+0.4,
        data_2005['urban_population'],
        color='olive',width=barWidth, label='2005')

plt.bar(np.arange(data_2010.shape[0])+0.5,
        data_2010['urban_population'],
        color='dodgerblue',width=barWidth, label='2010')

plt.bar(np.arange(data_2015.shape[0])+0.6,
        data_2015['urban_population'],
        color='slategray',width=barWidth, label='2015')



# show the legends on the plot
plt.legend()

# set the x-axis label
plt.xlabel('Country',fontsize=15)

# add title to the plot 
plt.title("Urban Population",fontsize=15)

# add countries names to the 11 groups on the x-axis
plt.xticks(np.arange(data_2005.shape[0])+0.2,
           ('United Arab Emirates', 'Armenia', 'Belgium', 'France',
       'Hong Kong SAR, China', 'Indonesia', 'Kenya', 'Nepal',
       'Saudi Arabia', 'Sweden', 'Eswatini', 'Tajikistan'),
             fontsize=10,rotation = 45)

# show the plot
plt.show()


# making dataframe of UAE data from the original dataframe
uae_df = orig_df[orig_df['country'] == 'United Arab Emirates']
uae_df.head(5)


# ### Implement a Function which removes Null values and return clean data


def remove_null_values(feature):
    return np.array(feature.dropna())


# ### For the Features Present In UAE DataFrame remove the null values 
# ### Print Each Features Size 

# Making dataframe of all the feature in the avaiable in 
# UAE dataframe passing it to remove null values function 
# for dropping the null values 
greenhouse = remove_null_values(uae_df[['greenhouse_gas_emissions']])

co2_emissions = remove_null_values(uae_df[['co2_emissions']])

argicultural_land = remove_null_values(uae_df[['agricultural_land']])

nitrous_oxide = remove_null_values(uae_df[['nitrous_oxide']])

fresh_water = remove_null_values(uae_df[['fresh_water']])

cereal_yield = remove_null_values(uae_df[['cereal_yield']])

arable_land = remove_null_values(uae_df[['arable_land']])

population = remove_null_values(uae_df[['population_growth']])

urban_pop = remove_null_values(uae_df[['urban_population']])

gdp = remove_null_values(uae_df[['GDP']])

# find the lenght of each feature size
# this will help us in creating dataframe 
# to avoid axis bound error in data frame creation
print('greenhouse Length = '+str(len((greenhouse)))) 
print('argicultural_land Length = '+str(len(argicultural_land))) 
print('nitrous_oxide  Length = '+str(len(nitrous_oxide))) 
print('co2_emissions Length = '+str(len(co2_emissions)))
print('fresh_water Length = '+str(len(fresh_water)))
print('cereal_yield Length = '+str(len(cereal_yield)))
print('population Length = '+str(len(population)))
print('urban_pop Length = '+str(len(urban_pop)))
print('gdp Length = '+str(len(gdp)))


# after removing the null values we will create datafram for UAE data
uae_data = pd.DataFrame({'GreenHouse Gases': [greenhouse[x][0] for x in range(30)],
                                 'Argicultural_land': [argicultural_land[x][0] for x in range(30)],
                                 'co2_emissions': [co2_emissions[x][0] for x in range(30)],
                                 'fresh_water': [fresh_water[x][0] for x in range(30)],
                                 'Nitrous Oxide': [nitrous_oxide[x][0] for x in range(30)],
                                 'Population': [population[x][0] for x in range(30)],
                                 'cereal_yield': [cereal_yield[x][0] for x in range(30)],
                                 'Urban_pop': [urban_pop[x][0] for x in range(30)],
                                 'GDP': [gdp[x][0] for x in range(30)],
                                })



uae_data.head(5) # call first 5 row 


# ### Correlation Heatmap of UAE


# create correlation matrix
corr_matrix = uae_data.corr()
plt.figure(figsize=(10,10))

# Plot the correlation matrix using imshow
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')

# Add labels and adjust the plot
plt.colorbar()
plt.title('Correlation Matrix Of UAE')
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation='vertical')
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)


# Show the plot
plt.show()


# ### Create a dataframe contain only  Saudi Arabia data

# making dataframe of Saudi Arabia data
sa_df = orig_df[orig_df['country'] == 'Saudi Arabia']


# ### Remove Null values from Features


# Making dataframe of all the feature in the avaiable in 
# SA dataframe passing it to remove null values function 
# for dropping the null values 
greenhouse = remove_null_values(sa_df[['greenhouse_gas_emissions']])

co2_emissions = remove_null_values(sa_df[['co2_emissions']])

argicultural_land = remove_null_values(sa_df[['agricultural_land']])

nitrous_oxide = remove_null_values(sa_df[['nitrous_oxide']])

fresh_water = remove_null_values(sa_df[['fresh_water']])

cereal_yield = remove_null_values(sa_df[['cereal_yield']])

arable_land = remove_null_values(sa_df[['arable_land']])

population = remove_null_values(sa_df[['population_growth']])

urban_pop = remove_null_values(sa_df[['urban_population']])

gdp = remove_null_values(sa_df[['GDP']])

# find the lenght of each feature size
# this will help us in creating dataframe 
# to avoid axis bound error in data frame creation
print('greenhouse Length = '+str(len((greenhouse)))) 
print('argicultural_land Length = '+str(len(argicultural_land))) 
print('nitrous_oxide  Length = '+str(len(nitrous_oxide))) 
print('co2_emissions Length = '+str(len(co2_emissions)))
print('fresh_water Length = '+str(len(fresh_water)))
print('cereal_yield Length = '+str(len(cereal_yield)))
print('population Length = '+str(len(population)))
print('urban_pop Length = '+str(len(urban_pop)))
print('gdp Length = '+str(len(gdp)))


# ### Create a new DataFrame for Saudi Arabia data contain no null values

# after removing the null values we will create datafram for SA data
sa_data = pd.DataFrame({'GreenHouse Gases': [greenhouse[x][0] for x in range(28)],
                                 'Argicultural_land': [argicultural_land[x][0] for x in range(28)],
                                 'co2_emissions': [co2_emissions[x][0] for x in range(28)],
                                 'fresh_water': [fresh_water[x][0] for x in range(28)],
                                 'Nitrous Oxide': [nitrous_oxide[x][0] for x in range(28)],
                                 'Population': [population[x][0] for x in range(28)],
                                 'cereal_yield': [cereal_yield[x][0] for x in range(28)],
                                 'Urban_pop': [urban_pop[x][0] for x in range(28)],
                                 'GDP': [gdp[x][0] for x in range(28)],
                                })


# ### Correlation Heatmap of  Saudi Arabia 

# create correlation matrix
corr_matrix = sa_data.corr()
plt.figure(figsize=(10,10))

# Plot the correlation matrix using imshow
plt.imshow(corr_matrix, cmap='Greens', interpolation='none')

# Add labels and adjust the plot
plt.colorbar()
plt.title('Correlation Matrix Of Saudi Arabia ')
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation='vertical')
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)


# Show the plot
plt.show()


# ### Make dataframe of Hong Kong SAR, China data from the original dataframe
# making dataframe of Hong Kong SAR, China data from the original dataframe
hk_df = orig_df[orig_df['country'] == 'Hong Kong SAR, China']
hk_df.head(5)


# ### For the Features Present In Hong Kong DataFrame remove the null values 
# ### Print Each Features Size 


# Making dataframe of all the feature in the avaiable in 
# Hong Kong dataframe passing it to remove null values function 
# for dropping the null values 
greenhouse = remove_null_values(hk_df[['greenhouse_gas_emissions']])

co2_emissions = remove_null_values(hk_df[['co2_emissions']])

argicultural_land = remove_null_values(hk_df[['agricultural_land']])

nitrous_oxide = remove_null_values(hk_df[['nitrous_oxide']])

fresh_water = remove_null_values(hk_df[['fresh_water']])

cereal_yield = remove_null_values(hk_df[['cereal_yield']])

arable_land = remove_null_values(hk_df[['arable_land']])

population = remove_null_values(hk_df[['population_growth']])

urban_pop = remove_null_values(hk_df[['urban_population']])

gdp = remove_null_values(hk_df[['GDP']])

# find the lenght of each feature size
# this will help us in creating dataframe 
# to avoid axis bound error in data frame creation
print('greenhouse Length = '+str(len((greenhouse)))) 
print('argicultural_land Length = '+str(len(argicultural_land))) 
print('nitrous_oxide  Length = '+str(len(nitrous_oxide))) 
print('co2_emissions Length = '+str(len(co2_emissions)))
print('fresh_water Length = '+str(len(fresh_water)))
print('cereal_yield Length = '+str(len(cereal_yield)))
print('population Length = '+str(len(population)))
print('urban_pop Length = '+str(len(urban_pop)))
print('gdp Length = '+str(len(gdp)))


# after removing the null values we will create datafram for Hong Kong data
hk_df = pd.DataFrame({
                                 'Argicultural_land': [argicultural_land[x][0] for x in range(31)],
                                 
                                 
                                 
                                 'Population': [population[x][0] for x in range(31)],
                                
                                 'Urban_pop': [urban_pop[x][0] for x in range(31)],
                                 'GDP': [gdp[x][0] for x in range(31)],
                                })



# create correlation matrix of Hong Kong
corr_matrix = hk_df.corr()
plt.figure(figsize=(10,10))

# Plot the correlation matrix using imshow
plt.imshow(corr_matrix, cmap='Blues', interpolation='none')

# Add labels and adjust the plot
plt.colorbar()
plt.title("Correlation Heatmap of Hong Kong")
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation='vertical')
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)


# Show the plot
plt.show()




# we want to see countries nitrous_oxide over the years
# we need to filter our original data frame to get specific fields
nit_data = orig_df[['country','year','nitrous_oxide']]

# drop the null values present in the dataset
nit_data = nit_data.dropna()


orig_df.country.unique()


# ### Filter the Data For All the Countries 



uae = nit_data[nit_data['country'] == 'United Arab Emirates']
arm = nit_data[nit_data['country']== 'Armenia']
bel =  nit_data[nit_data['country'] == 'Belgium'] 
fr = nit_data[nit_data['country'] == 'France'] 
hk = nit_data[nit_data['country'] == 'Hong Kong SAR, China'] 
ind = nit_data[nit_data['country'] == 'Indonesia'] 
ken = nit_data[nit_data['country'] ==  'Kenya'] 
nep = nit_data[nit_data['country'] == 'Nepal'] 
sa = nit_data[nit_data['country'] == 'Saudi Arabia'] 
swe = nit_data[nit_data['country'] ==  'Sweden'] 
esw = nit_data[nit_data['country'] ==  'Eswatini'] 
taj = nit_data[nit_data['country']== 'Tajikistan'] 


# set fig size
plt.figure(figsize=(10,10))

# set the line plot value on x-axis and y-axis by year and nitrous_oxide respectively
plt.plot(uae.year, uae.nitrous_oxide, '--',label='United Arab Emirates')
plt.plot(arm.year, arm.nitrous_oxide,'--',label='Armenia')
plt.plot(bel.year, bel.nitrous_oxide,'--',label='Belgium')
plt.plot(fr.year, fr.nitrous_oxide,'--',label='France')
plt.plot(hk.year, hk.nitrous_oxide,'--',label='Hong Kong SAR, China')
plt.plot(ind.year, ind.nitrous_oxide,'--',label='Indonesia')
plt.plot(ken.year, ken.nitrous_oxide,'--',label='Kenya')
plt.plot(nep.year, nep.nitrous_oxide,'--',label='Nepal')
plt.plot(sa.year, sa.nitrous_oxide,'--',label='Saudi Arabia')
plt.plot(swe.year, swe.nitrous_oxide,'--',label='Sweden')
plt.plot(esw.year, esw.nitrous_oxide,'--',label='Eswatini')
plt.plot(taj.year, taj.nitrous_oxide,'-',label='Tajikistan')

#Set the X-axis label and make it bold
plt.xlabel('Year',fontweight='bold')

#Set the Y-axis labe
plt.ylabel('Emission rate',fontweight='bold')

# set the title
plt.title("Nitrous Oxide")

# show the legends on the plot and place it on suitable position
plt.legend(bbox_to_anchor=(0.99,0.6),shadow=True)

#show the line plot
plt.show()


# we want to see countries greenhouse_gas_emissions over the years
# we need to filter our original data frame to get specific fields
ghg_data = orig_df[['country','year','greenhouse_gas_emissions']]

# drop the null values present in the dataset
ghg_data = ghg_data.dropna()


# ### Filter the Data For All the Countries 

uae = ghg_data[ghg_data['country'] == 'United Arab Emirates']
arm = ghg_data[ghg_data['country']== 'Armenia']
bel =  ghg_data[ghg_data['country'] == 'Belgium'] 
fr = ghg_data[ghg_data['country'] == 'France'] 
hk = ghg_data[ghg_data['country'] == 'Hong Kong SAR, China'] 
ind = ghg_data[ghg_data['country'] == 'Indonesia'] 
ken = ghg_data[ghg_data['country'] ==  'Kenya'] 
nep = ghg_data[ghg_data['country'] == 'Nepal'] 
sa = ghg_data[ghg_data['country'] == 'Saudi Arabia'] 
swe = ghg_data[ghg_data['country'] ==  'Sweden'] 
esw = ghg_data[ghg_data['country'] ==  'Eswatini'] 
taj = ghg_data[ghg_data['country']== 'Tajikistan'] 


# ### Line Plot of Greenhouse Gas Emissions


# set fig size
plt.figure(figsize=(10,10))

# set the line plot value on x-axis and y-axis by year and nitrous_oxide respectively
plt.plot(uae.year, uae.greenhouse_gas_emissions, '--',label='United Arab Emirates')
plt.plot(arm.year, arm.greenhouse_gas_emissions,'--',label='Armenia')
plt.plot(bel.year, bel.greenhouse_gas_emissions,'--',label='Belgium')
plt.plot(fr.year, fr.greenhouse_gas_emissions,'--',label='France')
plt.plot(hk.year, hk.greenhouse_gas_emissions,'--',label='Hong Kong SAR, China')
plt.plot(ind.year, ind.greenhouse_gas_emissions,'--',label='Indonesia')
plt.plot(ken.year, ken.greenhouse_gas_emissions,'--',label='Kenya')
plt.plot(nep.year, nep.greenhouse_gas_emissions,'--',label='Nepal')
plt.plot(sa.year, sa.greenhouse_gas_emissions,'--',label='Saudi Arabia')
plt.plot(swe.year, swe.greenhouse_gas_emissions,'--',label='Sweden')
plt.plot(esw.year, esw.greenhouse_gas_emissions,'--',label='Eswatini')
plt.plot(taj.year, taj.greenhouse_gas_emissions,'-',label='Tajikistan')

#Set the X-axis label and make it bold
plt.xlabel('Year',fontweight='bold')

#Set the Y-axis labe
plt.ylabel('Emission rate',fontweight='bold')

# set the title
plt.title("Greenhouse Gas emissions")

# show the legends on the plot and place it on suitable position
plt.legend(bbox_to_anchor=(0.99,0.6),shadow=True)

#show the line plot
plt.show()



