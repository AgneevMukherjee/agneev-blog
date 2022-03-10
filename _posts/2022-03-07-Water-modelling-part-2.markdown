---
layout: post
title:  "5. Modelling water bodies - 2"
date:   2022-03-07 12:00:00 +0100
category: ['data science', 'time series', 'environment', 'water']
---

Welcome back. The last time we saw the details of the methodology that I used to tackle the [Acea Smart Water Analytics](https://www.kaggle.com/c/acea-water-prediction/overview) competition. Now, let's see the actual modelling of the water bodies. While the notebook containing all the code (available [here](https://github.com/AgneevMukherjee/agneev-blog/blob/main/acea-submission-code.ipynb)) deals with all the waterbodies, the procedure is repetitive, and so we will look in detail in this post at only one representative water body. As you will see, there's plenty to unpack for that one!
{: style="text-align: justify"}

## Aquifer Petrignano  <a id="Petrignano"></a>

The water body I have selected to demonstrate the modelling here is Aquifer Petrignano, one of the four aquifers present in the dataset. As per the stated goal of the competition, this 'aquifer' model I built is applicable to each aquifer. Only the initial [data wrangling](https://expressanalytics.com/blog/what-is-data-wrangling-what-are-the-steps-in-data-wrangling/) portion requires some manual intervention, while the rest of the process is automated and identical for each aquifer.
{: style="text-align: justify"}

The competition [data page](https://www.kaggle.com/c/acea-water-prediction/data) describes the Petrignano aquifer as follows:
{: style="text-align: justify"}

> The wells field of the alluvial plain between Ospedalicchio di Bastia Umbra and Petrignano is fed by three underground aquifers separated by low permeability septa. The aquifer can be considered a water table groundwater and is also fed by the Chiascio river. The groundwater levels are influenced by the following parameters: rainfall, depth to groundwater, temperatures and drainage volumes, level of the Chiascio river.
{: style="text-align: justify"}

<br>
The reason I chose this aquifer to demonstrate is since it has all the column types associated with aquifers (rainfall, depth to groundwater, temperature, volume and hydrometry), and only two target variables, unlike some of the other water bodies. The methods for loading and visualising the data, filling in the missing values, and the various modelling steps are largely similar for all the water bodies, and so I will take the time to explain these in detail.
{: style="text-align: justify"}

<br>
## Data loading and visualisation  <a id="visual"></a>
The first thing to do is to load the dataset. For every water body I loaded the csv file in an identical fashion, to keep things general.
{: style="text-align: justify"}

{% highlight python %}

# Load the dataset
train = pd.read_csv('../input/acea-water-prediction/Aquifer_Petrignano.csv', parse_dates=['Date'], dayfirst=True)
# Create a copy of the original data, since the 'train' dataframe will be modified
train_orig = train.copy()

{% endhighlight %}

<br>
Let us now look at the start and end of the data, as well as the columns in the dataframe.

{% highlight python %}

train.head()
train.tail()
train.info()

{% endhighlight %}

<font size="2">
<div style="overflow:auto;">
<table border=1>
  <thead>
    <tr style="text-align: right;">
      <th></th>
       <th>Date</th>
       <th>Rainfall_Bastia_Umbra</th>
       <th>Depth_to_Groundwater_P24</th>
       <th>Depth_to_Groundwater_P25</th>
       <th>Temperature_Bastia_Umbra</th>
       <th>Temperature_Petrignano</th>
       <th>Volume_C10_Petrignano</th>
       <th>Hydrometry_Fiume_Chiascio_Petrignano</th>
    </tr>
  </thead>
  <tbody>
      <tr>
        <th>0</th>
        <td>2006-03-14</td>
        <td>NaN</td>
        <td>-22.48</td>
        <td>-22.18</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
      </tr>
      <tr>
        <th>1</th>
        <td>2006-03-15</td>
        <td>NaN</td>
        <td>-22.38</td>
        <td>-22.14</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
      </tr>
      <tr>
        <th>2</th>
        <td>2006-03-16</td>
        <td>NaN</td>
        <td>-22.25</td>
        <td>-22.04</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
      </tr>
      <tr>
        <th>3</th>
        <td>2006-03-17</td>
        <td>NaN</td>
        <td>-22.38</td>
        <td>-22.04</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
      </tr>
      <tr>
        <th>4</th>
        <td>2006-03-18</td>
        <td>NaN</td>
        <td>-22.60</td>
        <td>-22.04</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
        <td>NaN</td>
      </tr>
    </tbody>
  </table>
</div>
</font>

<br>

<font size="2">
<div style="overflow:auto;">
<table border=1>
  <thead>
    <tr style="text-align: right;">
      <th></th>
       <th>Date</th>
       <th>Rainfall_Bastia_Umbra</th>
       <th>Depth_to_Groundwater_P24</th>
       <th>Depth_to_Groundwater_P25</th>
       <th>Temperature_Bastia_Umbra</th>
       <th>Temperature_Petrignano</th>
       <th>Volume_C10_Petrignano</th>
       <th>Hydrometry_Fiume_Chiascio_Petrignano</th>
    </tr>
  </thead>
  <tbody>
      <tr>
        <th>5218</th>
        <td>2020-06-26</td>
        <td>0.0</td>
        <td>-25.68</td>
        <td>-25.07</td>
        <td>25.7</td>
        <td>24.5</td>
        <td>-29930.688</td>
        <td>2.5</td>
      </tr>
      <tr>
        <th>5219</th>
        <td>2020-06-27</td>
        <td>0.0</td>
        <td>-25.80</td>
        <td>-25.11</td>
        <td>26.2</td>
        <td>25.0</td>
        <td>-31332.960</td>
        <td>2.4</td>
      </tr>
      <tr>
        <th>5220</th>
        <td>2020-06-28</td>
        <td>0.0</td>
        <td>-25.80</td>
        <td>-25.19</td>
        <td>26.9</td>
        <td>25.7</td>
        <td>-32120.928</td>
        <td>2.4</td>
      </tr>
      <tr>
        <th>5221</th>
        <td>2020-06-29</td>
        <td>0.0</td>
        <td>-25.78</td>
        <td>-25.18</td>
        <td>26.9</td>
        <td>26.0</td>
        <td>-30602.880</td>
        <td>2.4</td>
      </tr>
      <tr>
        <th>5222</th>
        <td>2020-06-30</td>
        <td>0.0</td>
        <td>-25.91</td>
        <td>-25.25</td>
        <td>27.3</td>
        <td>26.5</td>
        <td>-31878.144</td>
        <td>2.4</td>
      </tr>
    </tbody>
  </table>
</div>
</font>

<br>
![Image_1](/agneev-blog/assets/img/img_5_1.png?raw=true){: width="500", height="100" }

We see that the data starts in March 2006 and ends in June 2020, and that there are 7 columns in all: 1 Date, 1 Rainfall, 2 Depth to Groundwater, 2 Temperature, 1 Volume and 1 Hydrometry. We also see that there are lots of missing values, especially at the start of the data - the Date column, which has an entry for each row, has 5223 non-null values, while some of the columns have only 4199. It is clear that missing value imputation will play a key part - we will discuss that in detail later. Before that though, let us make a list of the target columns - a step that obviously must be done manually.
{: style="text-align: justify"}

{% highlight python %}

targets = ['Depth_to_Groundwater_P24', 'Depth_to_Groundwater_P25']

{% endhighlight %}

<br>
Next, I decided to remove all the February 29 rows from the data, since otherwise there are issues later on in the modelling process, all years other than leap years having 365 days. Since a leap day only occurs once every 1461 days, there is minimal loss of data. The following cell removes the leap days via masking.
{: style="text-align: justify"}

{% highlight python %}

def is_leap_and_29Feb(s):
    return (s.Date.dt.year % 4 == 0) & \
           ((s.Date.dt.year % 100 != 0) | (s.Date.dt.year % 400 == 0)) & \
           (s.Date.dt.month == 2) & (s.Date.dt.day == 29)

mask = is_leap_and_29Feb(train)

train = train.loc[~mask]

{% endhighlight %}

<br>
OK, the next step is removing those rows at the start of the dataframe that do not have values for one or more of the target variables. Why is this needed? While it is not the case for Aquifer Petrignano, there are some datasets where target values are not available for several years at the beginning. Imputing values and using these rows for model training is incorrect, as the model will learn to predict the imputed data instead of fulfilling the aim of matching the actual data. I therefore wrote the cell below to find out the maximum length of NaNs before a valid value for any of the target columns, and remove all the dataframe rows before this value. Any missing values in the targets that occur later in the dataframe will be imputed. Doesn't this contradict what I just said about imputation being inappropriate? Well, the gaps in data in the middle are a lot less numerous than the large gaps in the beginning, and so imputing these is an acceptable trade-off here.
{: style="text-align: justify"}

{% highlight python %}

i = 0
nan_len = []
while i<len(targets):
    nan_len.append(train[f'{targets[i]}'].notna().idxmax())
    i += 1

train = train.iloc[max(nan_len):].reset_index(drop = True)

{% endhighlight %}

Let us now see how the data looks visually, which will help determine if any data wrangling other than missing value imputation is needed. I should note here that I used a light touch to data wrangling, only making any changes to the data where I felt it was both necessary and justified.
{: style="text-align: justify"}

The cell below creates separate dataframes for each type of variable (e.g. rainfall, volume, etc.), both for ease of plotting and because they come in handy later in the modelling process. A list of the columns corresponding to that variable type is also created for later use.
{: style="text-align: justify"}

{% highlight python %}

rain_df = train.filter(regex=("Rainfall.*"))
rains = rain_df.columns.tolist()
rain_df.insert(0, column = 'Date', value = train['Date'])

vol_df = train.filter(regex=("Volume.*"))
vols = vol_df.columns.tolist()
vol_df.insert(0, column = 'Date', value = train['Date'])

temp_df = train.filter(regex=("Temperature.*"))
temps = temp_df.columns.tolist()
temp_df.insert(0, column = 'Date', value = train['Date'])

depth_df = train.filter(regex=("Depth_to_Groundwater.*"))
depths = depth_df.columns.tolist()
depth_df.insert(0, column = 'Date', value = train['Date'])

hydro_df = train.filter(regex=("Hydrometry.*"))
hydros = hydro_df.columns.tolist()
hydro_df.insert(0, column = 'Date', value = train['Date'])

{% endhighlight %}

<br>
Now let us plot the different variables. You will notice that all the plot codes are contained inside a if statement that checks if the particular type of variable exists. If, for instance, there is no hydrometry term, the length of the hydrometry list created earlier will be 0, and the code will not run. This takes care of different datasets not having all the different variable types (a recurrent problem), ensuring that the model runs smoothly and automatically for every dataset.
{: style="text-align: justify"}

First, the rainfall:
{% highlight python %}

if len(rains)>0:
    i = 0
    while i<len(rains):
        if i == 0:
            ax1 = rain_df.plot.scatter(x = 'Date', y = f'{rains[i]}', color=colours[i])
        else:
            axes = rain_df.plot.scatter(x = 'Date', y = f'{rains[i]}', color=colours[i], ax = ax1)
        y_axis = ax1.yaxis
        y_axis.set_label_text('Rainfall (mm)')
        i += 1

{% endhighlight %}

![Image_2](/agneev-blog/assets/img/img_5_2.png?raw=true){: width="400", height="200" }

We can see that for the rainfall, the graph is as expected. Most of the days receive no rainfall, the average rainy day sees perhaps 10-20 mm of rain, while there are a few exceptional days that have values greater than 50 mm. Note the X-axis; the rainfall data is only available from 2009.
{: style="text-align: justify"}

Next, the 'volume':
<br>
{% highlight python %}

if len(vols)>0:
    i = 0
    while i<len(vols):
        if i == 0:
            ax1 = vol_df.plot.scatter(x = 'Date', y = f'{vols[i]}', color=colours[i])
        else:
            axes = vol_df.plot.scatter(x = 'Date', y = f'{vols[i]}', color=colours[i], ax = ax1)
        y_axis = ax1.yaxis
        y_axis.set_label_text('Volume (m$^3$)')
        i += 1

{% endhighlight %}

![Image_3](/agneev-blog/assets/img/img_5_3.png?raw=true){: width="400", height="200" }
<br>
The volume term refers to the amount of water taken from the drinking water treatment plant, a critical parameter in determining its health. Here, we see that most of the days had water extraction of roughly 30000 m<sup>3</sup>, with no clear trends visible.
{: style="text-align: justify"}

<br>
{% highlight python %}

if len(temps)>0:
    i = 0
    while i<len(temps):
        if i == 0:
            ax1 = temp_df.plot.scatter(x = 'Date', y = f'{temps[i]}', color=colours[i])
        else:
            axes = temp_df.plot.scatter(x = 'Date', y = f'{temps[i]}', color=colours[i], ax = ax1)
        y_axis = ax1.yaxis
        y_axis.set_label_text('Temperature ($^o$C)')
        i += 1

{% endhighlight %}

![Image_4](/agneev-blog/assets/img/img_5_4.png?raw=true){: width="400", height="200" }
<br>

The two temperature terms show a seasonal trend, with maximum temperatures of around 30 <sup>o</sup>C towards the middle of the year, and a minimum value of around 0 <sup>o</sup>C around the turn of the year. This is as expected; after all seasons are virtually defined by the rise and fall in temperatures. This seasonality will be used for imputing missing temperature data.
{: style="text-align: justify"}

<br>
{% highlight python %}

if len(depths)>0:
    i = 0
    while i<len(depths):
        if i == 0:
            ax1 = depth_df.plot.scatter(x = 'Date', y = f'{depths[i]}', color=colours[i])
        else:
            axes = depth_df.plot.scatter(x = 'Date', y = f'{depths[i]}', color=colours[i], ax = ax1)
        y_axis = ax1.yaxis
        y_axis.set_label_text('Depth to groundwater (m)')
        i += 1

{% endhighlight %}

![Image_5](/agneev-blog/assets/img/img_5_5.png?raw=true){: width="400", height="200" }
<br>
'Depth to groundwater' refers to the groundwater level measured in terms of distance from the ground floor as detected by a piezometer. As the target variables for aquifers are the depth to groundwater terms, this is obviously a very important variable category. The two depth to groundwater terms in this dataset follow each other very closely, and both show a highly haphazard pattern, with the values dropping and rising by large amounts at seemingly random intervals.
{: style="text-align: justify"}

<br>
{% highlight python %}

if len(hydros)>0:
    i = 0
    while i<len(hydros):
        if i == 0:
            ax1 = hydro_df.plot.scatter(x = 'Date', y = f'{hydros[i]}', color=colours[i])
        else:
            axes = hydro_df.plot.scatter(x = 'Date', y = f'{hydros[i]}', color=colours[i], ax = ax1)
        y_axis = ax1.yaxis
        y_axis.set_label_text('Hydrometry (m)')
        i += 1

{% endhighlight %}

![Image_6](/agneev-blog/assets/img/img_5_6.png?raw=true){: width="400", height="200" }
<br>
Finally, we come to the hydrometry term, which is also a measure of the groundwater level, but as determined by a hydrometric station rather than using a piezometer. We see from the figure that this variable is more stable than the depth to groundwater terms, and it exhibits a mild seasonal pattern, far less pronounced than for temperatures.
{: style="text-align: justify"}

<br>
## Data imputation  <a id="impute"></a>

As I said earlier, data imputation plays a major part in tackling this competition dataset. Different imputation methods are suitable for the different variable types, as will be seen below. An additional point is that the imputation should occur not just for missing values, but also to replace erroneous data, if these can be definitively identified.
{: style="text-align: justify"}

Let us start with rainfall. For rainfalls, since 0 is a legitimate value, and is in fact the [mode](https://en.wikipedia.org/wiki/Mode_(statistics)) of the data, and since there is no theoretical upper limit to the amount of rain received, the only way any data can be identified as wrong is in the case of negative values or values perhaps a hundred times greater than the rest. This is not the case for any of the datasets, and hence we can focus simply on imputing the missing data. Using the monthly average of the rainfall may be one possible method, but as we saw in the rainfall graph above, there is no seasonality in the data, and hence using monthly averages to impute long stretches of time will probably bias the data. As per the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle), complexity should be avoided unless necessary. In the case of rainfall, inserting the mode value of 0 (i.e. no rainfall on that day) in place of the missing data is the simplest solution, and in the absence of any better alternatives, I decided to use this.
{: style="text-align: justify"}

{% highlight python %}

if len(rains)>0:
    i = 0
    while i<len(rains):
        train[f'{rains[i]}'] = train[f'{rains[i]}'].fillna(0)
        i += 1

{% endhighlight %}

Next, the volume. As mentioned above, the volume term refers to the amount of water taken from the drinking water treatment plant. It can therefore have a value of 0 if no water is taken on that day. The same points as made for rainfall also hold for volume, and therefore I did the imputation the same way too.
{: style="text-align: justify"}

{% highlight python %}

if len(vols)>0:
    i = 0
    while i<len(vols):
        train[f'{vols[i]}'] = train[f'{vols[i]}'].fillna(0)
        i += 1

{% endhighlight %}


We have seen that the temperature terms show a seasonal trend as expected, and I used this for imputation. The invalid data is a little hard to identify here, given that the data correctly has positive, negative and zero values. Again, values far outside the range of the rest of the data may be erroneous, and a statistical measure like the [z-score](https://en.wikipedia.org/wiki/Standard_score) may be used to identify these outliers. An inspection of the minimum and maximum values of the temperature data for all the datasets did not turn up anything untoward, and hence I decided to skip further statistical inspection. Another source of erroneous data, however, does appear to exist - long stretches of time with values of 0, likely indicating corrupt data. Somewhat arbitrarily, I decided that 7 or more consecutive values of 0 were errors, and decided to mask and replace these with NaNs for imputation.
{: style="text-align: justify"}

{% highlight python %}

if len(temps)>0:
    i = 0
    while i<len(temps):
        a = train[f'{temps[i]}'] == 0
        mask = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0) >= 7
        train[f'{temps[i]}'] = train[f'{temps[i]}'].mask(mask, np.nan)
        temp_df[f'{temps[i]}'] = temp_df[f'{temps[i]}'].mask(mask, np.nan)
        i += 1

# Replacing 7 or more 0 temperature values with Nans
# Taken from
#https://stackoverflow.com/questions/42946226/replacing-more-than-n-consecutive-values-in-pandas-dataframe-column   

{% endhighlight %}

<br>
That done, we can now turn to the actual data imputation, which is done in the cell below.

{% highlight python %}

if len(temps)>0:
    i = 0
    while i<len(temps):
        X = pd.DataFrame(temp_df.iloc[:, np.r_[0,i+1]])
        X['dayofyear'] = X['Date'].dt.dayofyear
        X2 = X.groupby('dayofyear').mean()
        X2.reset_index(level=0, inplace=True)
        X.loc[X[f'{temps[i]}'].isnull() == True, 'same_day'] =\
        X['dayofyear'].map(X2.set_index('dayofyear')[f'{temps[i]}'].to_dict())
        X.loc[X[f'{temps[i]}'].isnull() == False, 'same_day'] = X[f'{temps[i]}']
        X.loc[X[f'{temps[i]}'].isnull() == True, f'{temps[i]}' + '_new'] = X['same_day']
        X.loc[X[f'{temps[i]}'].isnull() == False, f'{temps[i]}' + '_new'] = X[f'{temps[i]}']

        train[f'{temps[i]}'].update(X[f'{temps[i]}' + '_new'])

        i += 1

    i = 0
    while i<len(temps):
        if i == 0:
            ax1 = train.plot.scatter(x = 'Date', y = f'{temps[i]}', color=colours[i])
        else:
            axes = train.plot.scatter(x = 'Date', y = f'{temps[i]}', color=colours[i], ax = ax1)
        y_axis = ax1.yaxis
        y_axis.set_label_text('Temperature ($^o$C)')
        i += 1

{% endhighlight %}

Briefly, the average temperature for each day of the year is calculated, and this average is used to fill in missing data for that date. For example, if the temperature value for June 29, 2018 is missing, then the average temperature of June 29 calculated using the rest of the data will be filled in. As we can see from the figures showing the original (left) and the imputed data (right) below, this method works very well. In the original temperature data, you see that the 'red' graph has a big stretch of zeroes around 2015, and the updated graph looks much more reasonable in this timespan.
{: style="text-align: justify"}

![Image_4](/agneev-blog/assets/img/img_5_4.png?raw=true){: width="350", height="200" }
![Image_7](/agneev-blog/assets/img/img_5_7.png?raw=true){: width="350", height="200" }

<br>
Next, let us turn to the hydrometry term, which also has a weak seasonal pattern, and hence can be imputed similarly to temperature. Unlike all the previous items, however, the 0 values here _can_ be considered as errors - the values cannot suddenly depart from their usual 2-4 m range and become 0. Therefore, I replaced all the 0 values by NaNs. I made a fresh dataframe for the hydro terms from the imputed data, for reasons that I will explain when I touch upon Aquifer Auser. The rest of the procedure is the same as for temperature. The figure below shows that this imputation method gives adequate results for hydrometry, though less realistic than was the case for temperature.
{: style="text-align: justify"}

{% highlight python %}

if len(hydros)>0:
    i = 0
    while i<len(hydros):
        train[f'{hydros[i]}'].replace(0, np.nan, inplace=True)
        i += 1
# 0s in hydros are errors, so replace them with Nans
    hydro_df2 = train.filter(regex=("Hydrometry.*"))
    hydros = hydro_df2.columns.tolist()
    hydro_df2.insert(0, column = 'Date', value = train['Date'])

{% endhighlight %}

![Image_6](/agneev-blog/assets/img/img_5_6.png?raw=true){: width="350", height="200" }
![Image_8](/agneev-blog/assets/img/img_5_8.png?raw=true){: width="350", height="200" }

That leaves only one more term to deal with. I have kept the depth of groundwater term for the last, since this is easily the most difficult to impute. Going over all the different datasets, no one method, e.g. mean, rolling mean, fill forward, K-nearest neighbours, seasonal interpolation, etc. worked well for every depth of groundwater series. This left me with two alternatives - either use a bespoke imputation method for each depth of groundwater after manual inspection and analysis, or use a [multiple imputation method](https://www.bmj.com/content/338/bmj.b2393). Multiple imputation methods compensate for the uncertainty in the missing data by creating and combining predictions from several plausible imputed datasets. Their use permits automatic processing, which was one of my aims, and hence I decided to use these for imputing the depth of groundwater data. The question, though, was: which of the several different multiple imputation methods should I use?
{: style="text-align: justify"}

Again, there is no perfect solution, but some [studies](https://bmjopen.bmj.com/content/3/8/e002847)/[opinions](https://www.analyticsvidhya.com/blog/2016/03/tutorial-powerful-packages-imputing-missing-values/) show that the RF-based MissForest algorithm (details can be found [here](https://towardsdatascience.com/missforest-the-best-missing-data-imputation-algorithm-4d01182aed3) or [here](https://www.kaggle.com/lmorgan95/missforest-the-best-imputation-algorithm)) can outperform alternatives like MICE and Hmisc. We will therefore use this method. The original MissForest algorithm was implemented in R, and sklearns' IterativeImputer, which [has an ExtraTreesRegressor estimator](https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html) similar to MissForest is [currently experimental](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer) and not stable. I therefore used the [missingpy module](https://pypi.org/project/missingpy/). I carried out the imputation on a copy of the original dataset to avoid the fitting being influenced by the imputations of the other variables carried out above. Like with the hydrometry term, I treated 0 values as missing data. As the algorithm cannot handle 'Date' terms, and they are not necessary for the imputation anyway, I dropped the Date column prior to the run, and then added it back to the imputed dataframe.
{: style="text-align: justify"}

{% highlight python %}

imputer = MissForest()
train_copy = train_orig.copy()
if len(depths)>0:
    i = 0
    while i<len(depths):
        train_copy[f'{depths[i]}'].replace(0, np.nan, inplace=True)
        i += 1
# 0s in depths are errors, so replace them with Nans
    train_copy = train_copy.drop(['Date'], axis=1)
    col_list = list(train_copy)
    imputs = imputer.fit_transform(train_copy)
    imputs = pd.DataFrame(data=imputs, columns=col_list)
    imputs['Date']= train_orig['Date']

{% endhighlight %}

The imputed values are used to replace the missing values in the cell below:

{% highlight python %}

if len(depths)>0:
    i = 0
    while i<len(depths):
        X = pd.DataFrame(depth_df.iloc[:, np.r_[0,i+1]])
        X.loc[X[f'{depths[i]}'].isnull() == True, f'{depths[i]}' + '_new'] =\
        X['Date'].map(imputs.set_index('Date')[f'{depths[i]}'].to_dict())
        X.loc[X[f'{depths[i]}'].isnull() == False, f'{depths[i]}' + '_new'] = X[f'{depths[i]}']
        train[f'{depths[i]}'] = X[f'{depths[i]}' + '_new']
        i += 1
    i = 0
    while i<len(depths):
        if i == 0:
            ax1 = train.plot.scatter(x = 'Date', y = f'{depths[i]}', color=colours[i])
        else:
            axes = train.plot.scatter(x = 'Date', y = f'{depths[i]}', color=colours[i], ax = ax1)
        y_axis = ax1.yaxis
        y_axis.set_label_text('Depth to groundwater (m)')
        i += 1

{% endhighlight %}

It must be said that the plot showing the imputed data (below, right) is not the best advertisement for MissForest, as there are some fitted points above and below the actual values. Overall, though, I found MissForest to be the most reliable and generalisable of all the imputation methods I tried on all the different depth of groundwater series, and hence decided to stick with it.
{: style="text-align: justify"}

![Image_5](/agneev-blog/assets/img/img_5_5.png?raw=true){: width="350", height="200" }
![Image_9](/agneev-blog/assets/img/img_5_9.png?raw=true){: width="350", height="200" }

<br>
## Forecast period  <a id="Period"></a>

After concluding the data inspection and imputation, we are finally ready to look at the actual models. The first thing I had to decide is the time frame for which we want the predictions, i.e. how many days ahead do we want to predict. There was plenty of discussion on this topic in the competition forums, and it was stated by the organisers that the prediction period is difficult to predetermine, but needs to be decided on a case-by-case basis depending on the water body type and other factors. I therefore created a variable called days_ahead that allows the user to specify the desired time period, the model automatically forecasting for that period. The default value I used is 7 days, but I also ran it for 30 days for this waterbody to show how it works and how the predictions are affected by the forecast period.
{: style="text-align: justify"}

{% highlight python %}

days_ahead = 7

target_cols = targets.copy()
i = 0
for target in targets:
    train[target + '_target'] = train[target].shift(-days_ahead)
    target_cols[i] = target + '_target'
    i += 1
# Creating new columns by shifting the target columns by the forecast period
train = train.dropna()
# Required as the shift in the cell above introduces some new NaNs

{% endhighlight %}

<br>
## Random forest model  <a id="RF"></a>

Let us now look at a Random Forest (RF) model. The first thing is to make a copy of the train dataframe that we have carried out all the data wrangling on, since that dataframe will also be used for the LGBM and LSTM models. The next thing is to carry out the feature engineering needed for a tree-based model to make predictions. This involves calculating the rolling mean for the variables where the cumulative sum is meaningful for modelling (i.e. rainfall and volume - the sum of the daily rainfall over a week can be an important input to the model) and the lag terms where this sum is meaningless (i.e. temperatures, hydrometry and depth of gorundwater - the 'sum' of the daily temperature over a week has no utility as such). Based on preliminary tests, separate rolling mean terms for every 30 days till 180 days and four lag terms till 90 days work sufficiently well without being either an excessive or an insufficient number of terms, and hence these are calculated in the cell below.
{: style="text-align: justify"}

{% highlight python %}

train_rf = train.copy()

if len(rains)>0:
    for r in rains:
        train_rf[r + '_roll_030'] = train_rf[r].rolling(30).mean().astype('float32')
        train_rf[r + '_roll_060'] = train_rf[r].rolling(60).mean().astype('float32')
        train_rf[r + '_roll_090'] = train_rf[r].rolling(90).mean().astype('float32')
        train_rf[r + '_roll_120'] = train_rf[r].rolling(120).mean().astype('float32')
        train_rf[r + '_roll_150'] = train_rf[r].rolling(150).mean().astype('float32')
        train_rf[r + '_roll_180'] = train_rf[r].rolling(180).mean().astype('float32')
if len(vols)>0:
    for v in vols:
        train_rf[v + '_roll_030'] = train_rf[v].rolling(30).mean().astype('float32')
        train_rf[v + '_roll_060'] = train_rf[v].rolling(60).mean().astype('float32')
        train_rf[v + '_roll_090'] = train_rf[v].rolling(90).mean().astype('float32')
        train_rf[v + '_roll_120'] = train_rf[v].rolling(120).mean().astype('float32')
        train_rf[v + '_roll_150'] = train_rf[v].rolling(150).mean().astype('float32')
        train_rf[v + '_roll_180'] = train_rf[v].rolling(180).mean().astype('float32')
if len(temps)>0:
    for t in temps:
        train_rf[t + '_week_lag'] = train_rf[t].shift(7).astype('float32')
        train_rf[t + '_month_lag'] = train_rf[t].shift(30).astype('float32')
        train_rf[t + '_bimonth_lag'] = train_rf[t].shift(60).astype('float32')
        train_rf[t + '_quarter_lag'] = train_rf[t].shift(90).astype('float32')
if len(hydros)>0:
    for h in hydros:
        train_rf[h + '_week_lag'] = train_rf[h].shift(7).astype('float32')
        train_rf[h + '_month_lag'] = train_rf[h].shift(30).astype('float32')
        train_rf[h + '_bimonth_lag'] = train_rf[h].shift(60).astype('float32')
        train_rf[h + '_quarter_lag'] = train_rf[h].shift(90).astype('float32')
if len(depths)>0:
    for d in depths:
        train_rf[d + '_week_lag'] = train_rf[d].shift(7).astype('float32')
        train_rf[d + '_month_lag'] = train_rf[d].shift(30).astype('float32')
        train_rf[d + '_bimonth_lag'] = train_rf[d].shift(60).astype('float32')
        train_rf[d + '_quarter_lag'] = train_rf[d].shift(90).astype('float32')

{% endhighlight %}

If the non-shifted target terms are kept in the dataframe, then the RF focusses on using those to predict the shifted target terms, which is contrary to our aim of getting it to predict based on the input variables. These are therefore dropped in the next cell. Alongside, the lag terms introduce NaNs at the start of the dataframe which will also be removed in the following cell in the same way we saw earlier. The dropna removes all rows containing NaN, and the index[0] gives the first row of the resulting dataframe.
{: style="text-align: justify"}

{% highlight python %}

i = 0
for target in targets:
    train_rf = train_rf.drop([f'{targets[i]}'], axis=1)
    i +=1
# If the target columns are not removed, the RF just focuses on them.
train_rf = train_rf.loc[train_rf.dropna().index[0]:].reset_index(drop = True)

{% endhighlight %}

Now we are ready to split the dataframe into the train and test sets. Since we are trying to see how good our model is at predicting the future, the final year of data, i.e. from 1/7/2019 to 30/6/2020 will be used as the test set, with all the earlier data being used for training. We will further split the data into X (input) and y (output) terms. All this is done in the following cell.
{: style="text-align: justify"}

{% highlight python %}

rf_X = train_rf[train_rf.columns.difference(target_cols)]
rf_y = train_rf[target_cols]
train_rf_X = rf_X.iloc[:rf_X.loc[rf_X['Date'] == '2019-07-01'].index.item()].reset_index(drop = True)
test_rf_X = rf_X.iloc[rf_X.loc[rf_X['Date'] == '2019-07-01'].index.item():].reset_index(drop = True)
train_rf_y = rf_y.iloc[:rf_y.loc[rf_X['Date'] == '2019-07-01'].index.item()].reset_index(drop = True)
test_rf_y = rf_y.iloc[rf_y.loc[rf_X['Date'] == '2019-07-01'].index.item():].reset_index(drop = True)
# Note that rf_y does not have a Date column, but as it's the same length as rf_X,
# the Date column from that df can be used for the train-test split
col_list2 = list(train_rf.columns)
train_rf_X = train_rf_X.drop(['Date'], axis=1)
test_rf_X = test_rf_X.drop(['Date'], axis=1)
# The Date column needs to be removed for the RF regressor to work

{% endhighlight %}

<br>
Now, I want to clarify one thing to prevent any misgivings on the issue. The rolling and lag terms for the first few items of the test set will be based on data from the end of the training set. Is this data leakage? No.
{: style="text-align: justify"}

Data leakage refers to ['when the data you are using to train a machine learning algorithm happens to have the information you are trying to predict'](https://insidebigdata.com/2014/11/26/ask-data-scientist-data-leakage/). This is not the case here, as we are not trying to predict any of the rolling or lag terms. Further, we can see in the above link that leakage occurs due to 'leaking data from the test set into the training set', 'leaking the correct prediction or ground truth into the test data', 'leaking of information from the future into the past' and 'information from data samples outside of scope of the algorithm’s intended use'. None of these is the case here.
{: style="text-align: justify"}

In short, leakage only occurs if information that will not be available at prediction time is used in model training. All the lag and rolling means terms are in the past and thus will be available at prediction time, and hence this is not a data leak.
{: style="text-align: justify"}

<br>
Having clarified the above point, we can now run the RF regressor. I maintained the default number of estimators (100), but this can be increased or decreased depending on whether accuracy and robustness or processing time is of greater importance. The max_depth item I kept at 30 to prevent overfitting, while I passed a random_state int value of 2 to [ensure consistency across calls](https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn). All the other (hyper)parameters were kept at their default. Hyperparameter tuning may lead to somewhat improved accuracy at the cost of increased complexity, run time and perhaps overfitting, and if desired, can be investigated further as per online guides, such as [this](https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/) or [this](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d).
{: style="text-align: justify"}

{% highlight python %}

regr_rf = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=2)
regr_rf.fit(train_rf_X, train_rf_y)

{% endhighlight %}

<br>
After the initial fit carried out above, it is better to take the training features deemed most important as per permutation importance and refit the RF only on these. This will enable the RF to focus on these important features, rather than spreading its attention, so to speak, among a lot of different features of marginal importance. This is done below. The code is mostly self-explanatory, but [this reference](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb) may be used for better understanding certain portions if needed.
{: style="text-align: justify"}

{% highlight python %}

# Calculating permutation importance based on the previous RF
train_val_rf = permutation_importance(regr_rf, train_rf_X, train_rf_y, n_repeats=10,
                                random_state=42, n_jobs=-1)
sorted_idx_val = train_val_rf.importances_mean.argsort()

# Taking the important features from the random forest above
var_names = train_rf_X.columns[sorted_idx_val].to_frame()
imps = sorted(train_val_rf.importances_mean)
var_names = var_names.assign(imps = imps)
var_names = var_names.rename(columns={0: "vars"})

# Making a list containing the most important variables. The feature importance value of 0.001
# is arbitrary and can be changed if needed
to_keep = var_names[var_names.imps>0.001].vars.to_list()

# Making new train and test dataframes for the X terms judged important enough to keep
train_rf_X = train_rf_X[[c for c in train_rf_X.columns if c in to_keep]]
test_rf_X = test_rf_X[[c for c in test_rf_X.columns if c in to_keep]]

{% endhighlight %}

<br>
We can now rerun the RF on the new dataframe containing the most important features and obtain the predictions. We can then have a look at the features now deemed most important as per permutation importance, this time using the test data instead of the training data as above.
{: style="text-align: justify"}

{% highlight python %}

regr_rf.fit(train_rf_X, train_rf_y)
rf_preds_1 = regr_rf.predict(test_rf_X)

result_val2 = permutation_importance(regr_rf, test_rf_X, test_rf_y, n_repeats=10,
                                random_state=42, n_jobs=-1)
sorted_idx_val2 = result_val2.importances_mean.argsort()

fig1, ax = plt.subplots(figsize=(10,5))
ax.boxplot(result_val2.importances[sorted_idx_val2].T,
           vert=False, labels=test_rf_X.columns[sorted_idx_val2])
ax.set_title("Permutation Importances (test set)")
fig1.tight_layout()
plt.show()

{% endhighlight %}

![Image_10](/agneev-blog/assets/img/img_5_10.png?raw=true){: width="800", height="400" }

We can see that the weekly lag terms for the two targets are the most important predictors, which is unsurprising. Among the other inputs, the 30-day rolling mean of the volume term and the 120-day rolling mean of the rainfall term are of relatively major importance. Neither of the temperature terms appears in the list of important features at all, perhaps because they had been dropped when we narrowed down the list of features earlier.
{: style="text-align: justify"}

Finally, we can calculate the MAE of the predictions. The predictions appear to be reasonably accurate - given that the depth to groundwater terms are in the range of 19 and 35 m, an average absolute error of around 0.3 m is a little over 1%. The RMSE values are also provided below for reference, and are slightly higher than the MAE numbers.
{: style="text-align: justify"}

{% highlight python %}

mean_absolute_error(rf_preds_1, test_rf_y, multioutput='raw_values')

mean_squared_error(rf_preds_1, test_rf_y, multioutput='raw_values', squared=False)

{% endhighlight %}

![Image_11](/agneev-blog/assets/img/img_5_11.png?raw=true){: width="300", height="200" }

<br>
We have looked at the most important features in terms of permutation importance above. Let us now see what the SHAP values calculated using SHAP's TreeExplainer say. First a look at the summary plot for all the target columns combined:
{: style="text-align: justify"}

![Image_12](/agneev-blog/assets/img/img_5_12.png?raw=true){: width="600", height="300" }

We see that the weekly lag of the two depth to groundwater terms are important predictors for both the target terms, in line with both what the permutation importances showed and our expectations.The 120-day rolling mean of the rainfall is also an important predictor, more for the Depth_to_Groundwater_P24 target than Depth_to_Groundwater_P25 (the blue portion of the bar is longer than the red portion for the Rainfall_Bastia_Umbra_roll_120 term). Three different rolling means of the volume term are of moderate importance, while the weekly lag of the hydrometry term comes much lower down than was the case for the permutation importances. Overall, there is relatively good agreement, but the SHAP plots can provide greater depth of information, as shown below.
{: style="text-align: justify"}

<br>
Now let us see what the individual SHAP summary plots show.

{% highlight python %}

i = 0
for target in targets:
    # To ensure the code works if only target variable is present
    if len(targets) == 1:
        shap.summary_plot(shap_values, test_rf_X, show=False)
    else:
        shap.summary_plot(shap_values[i], test_rf_X, show=False)
    plt.title(f'{targets[i]}', fontsize=20)
    plt.show()
    i +=1

{% endhighlight %}

![Image_13](/agneev-blog/assets/img/img_5_13.png?raw=true){: width="600", height="300" }

![Image_14](/agneev-blog/assets/img/img_5_14.png?raw=true){: width="600", height="300" }

A detailed explanation of how to interpret SHAP summary plots can be found [here](https://christophm.github.io/interpretable-ml-book/shap.html#shap-summary-plot), but looking at the above, it is clear that they present some interesting insights. Consider the Depth_to_Groundwater_P24 figure. We see that the Depth_to_Groundwater_P24_week_lag term has a positive impact on the model output (except for very low values of the feature), while the Depth_to_Groundwater_P25_week_lag has a negative impact (except for very high values of the feature. Thus, we see that very high values of the Depth_to_Groundwater_P24_week_lag (bright red colour) leads to the model output increasing by around 1.2, while for the Depth_to_Groundwater_P25_week_lag term, very high values affect the model output by -0.2 to +0.1. Very low values of this feature (bright blue) reduce the model output by around 1.5. We also see that the Rainfall_Bastia_Umbra_roll_120 term has a small postive effect, but this effect is a bit inconsistent, as a mixture of blue and red values are seen in the 0.1 to 0.5 region. Finally, all the other terms are centred around 0 with limited spread on either side, indicating that it does not much matter to the model predictions whether these terms have a high or low value - in other words, that they are of low importance in prediction.
{: style="text-align: justify"}

The above thus helps us not only see which features are important, but also how they actually impact the model - whether this impact is positive or negative, and whether high values and low values of the feature affect predictions differently. While the above can be discussed in further depth, in the interests of brevity, let us move on to the LGBM model.
{: style="text-align: justify"}

<br>
## LightGBM model  <a id="LGBM"></a>

First things first - the LightGBM algorithm [does not support multi-output regression](https://github.com/microsoft/LightGBM/issues/524), and [neither does the competing library XGBoost](https://github.com/dmlc/xgboost/issues/2087), for that matter. This hurdle can be tackled in two ways. One is to use sklearns' [MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html) as a wrapper. This works very well indeed, but in our case presents one hitch - SHAP does not support MultiOutputRegressor. This brings us to the second alternative - run LGBM for each target term. This would normally be an inferior alternative, as the interactions between the target terms would not be captured. In this case, for instance, if we run the LGBM for the Depth_to_Groundwater_P24 term, dropping the Depth_to_Groundwater_P25 term, and then the other way round, the effect of the Depth_to_Groundwater_P24 term on the Depth_to_Groundwater_P25 term and vice versa would not be accounted for in the model. Here, though, we are using the lag terms of the targets, which are present during the modelling step, and hence the target variables' interactions _are_ captured in the model. We will therefore use the LGBMRegressor in a for-loop for each of the target variables. We will stick to the default LightGBM parameters in the interest of time, simplicity, and robustness, but more information about parameter tuning can be found [here](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) and [here](https://neptune.ai/blog/lightgbm-parameters-guide).
{: style="text-align: justify"}

The train and test sets are prepared in the same way as for RF, reusing the lag and rolling mean terms we had calculated for using in the RF. As a Gradient Boosting method works sequentially, [progressively improving the model to minimise prediction error](https://www.displayr.com/gradient-boosting-the-coolest-kid-on-the-machine-learning-block/), a feature selection step as was carried out for the RF is less important - less important features will anyway be dropped as the algorithm finetunes the predictions. In theory, a lower number of features should allow for lower run time, but the feature selection itself is time consuming, and so I did not carry this out for the LGBM model. Therefore, all we now need to do before running the regressor is to drop the Date columns, and create an empty numpy array to hold the model predictions.
{: style="text-align: justify"}

{% highlight python %}

regr_lgbm = LGBMRegressor()

train_lgbm_X = rf_X.iloc[:rf_X.loc[rf_X['Date'] == '2019-07-01'].index.item()].reset_index(drop = True)
test_lgbm_X = rf_X.iloc[rf_X.loc[rf_X['Date'] == '2019-07-01'].index.item():].reset_index(drop = True)
train_lgbm_y = rf_y.iloc[:rf_y.loc[rf_X['Date'] == '2019-07-01'].index.item()].reset_index(drop = True)
test_lgbm_y = rf_y.iloc[rf_y.loc[rf_X['Date'] == '2019-07-01'].index.item():].reset_index(drop = True)
# Note that rf_y does not have a Date column, but as it's the same length as rf_X,
# the Date column from that df can be used for the train-test split
train_lgbm_X = train_lgbm_X.drop(['Date'], axis=1)
test_lgbm_X = test_lgbm_X.drop(['Date'], axis=1)

lgbm_preds_1 = np.empty_like(test_lgbm_y)

{% endhighlight %}

{% highlight python %}

i = 0
for target in targets:
    m1 = regr_lgbm.fit(train_lgbm_X, train_lgbm_y.iloc[:,i])
    lgbm_preds_1[:,i] = m1.predict(test_lgbm_X)
    explainer2 = shap.TreeExplainer(regr_lgbm)
    shap_values = explainer2.shap_values(test_lgbm_X)
    shap.summary_plot(shap_values, test_lgbm_X, show=False)
    plt.title(f'{targets[i]}', fontsize=20)
    plt.show()
    i +=1

{% endhighlight %}

![Image_15](/agneev-blog/assets/img/img_5_15.png?raw=true){: width="600", height="300" }

![Image_16](/agneev-blog/assets/img/img_5_16.png?raw=true){: width="600", height="300" }

The permutation importances can be calculated for the LGBM similarly to RF, but let us focus on the SHAP summary plots. We see that these are broadly similar to the RF plots, with some differences. In the Depth_to_Groundwater_P24 plot, we see that the impact of the Depth_to_Groundwater_P24 weekly lag term is less pronounced than was the case for RF, with high values increasing model output by less than 1.0. We also see the individual rainfall terms making less of an impact on the model output. These differences are partly a result of the differences in the algorithm, and partly simply due to the greater number of features available to the LGBM regressor, since we did not carry out a feature selection for it. The broad picture, though, remains the same - the weekly lag terms for the depths to groundwater are the most important, the volume and rainfall terms have a part to play, and the hydrometry and temperature terms have a negligible impact on the model output.
{: style="text-align: justify"}

Looking at the accuracy of the predictions, we see numbers fairly similar to those obtained for RF.
{: style="text-align: justify"}

{% highlight python %}

mean_absolute_error(lgbm_preds_1, test_lgbm_y, multioutput='raw_values')

mean_squared_error(lgbm_preds_1, test_lgbm_y, multioutput='raw_values', squared=False)

{% endhighlight %}

![Image_17](/agneev-blog/assets/img/img_5_17.png?raw=true){: width="300", height="200" }

<br>
## LSTM model  <a id="LSTM"></a>

Finally, we come to the LSTM model. The train-test split I did similarly to the RF and LGBM, but the split into the X and y arrays I did as per the code given [here](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/). In fact, the LSTM model I used is itself adapted from the 'LSTM Model With Univariate Input and Vector Output' section of that page, and hence the explanations given there are applicable here. The only item that needs separate explanation is the scaling step. Scaling is [very useful](https://stackoverflow.com/questions/46686924/why-scaling-data-is-very-important-in-neural-networklstm/46688787) in helping NNs like LSTM make quicker and more accurate predictions, but needs to be carried out [post the train-test split](https://stackoverflow.com/questions/49444262/normalize-data-before-or-after-split-of-training-and-testing-data) to avoid data leakage, which is what has been done below. Sklearn's [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) has been used for scaling.
{: style="text-align: justify"}

{% highlight python %}

# Make a copy of the train dataframe
train_lstm = train.copy()
train_lstm = train_lstm.loc[train_lstm.dropna().index[0]:].reset_index(drop = True)
# Create train-test split, then drop Date columns
train_train_lstm = train_lstm.iloc[:train_lstm.loc[train_lstm['Date'] == '2019-07-01'].index.item()]
train_test_lstm = train_lstm.iloc[train_lstm.loc[train_lstm['Date'] == '2019-07-01'].index.item():]
train_train_lstm = train_train_lstm.drop(['Date'], axis=1)
train_test_lstm = train_test_lstm.drop(['Date'], axis=1)
# Create a copy of the train test dataframes and perform scaling on thiese dataframes
# Separate scalers used for the train and test sets to prevent the possibility of data leakage
train_train_lstm2 = train_train_lstm.copy()
train_test_lstm2 = train_test_lstm.copy()
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
train_train_lstm2 = scaler1.fit_transform(train_train_lstm2)
train_test_lstm2 = scaler2.fit_transform(train_test_lstm2)

{% endhighlight %}

The function below splits the train and test sets into separate arrays containing the X and y terms, and it is run to obtain the inputs and outputs in the shape required for the LSTM. As mentioned above, the portion below is based on the code [here](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/).
{: style="text-align: justify"}

{% highlight python %}

# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-len(target_cols)], sequences[end_ix-1, -len(target_cols):]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Choose number of time steps used for training. Arbitrary, 7 chosen here.
n_steps = 7

# Convert into input/output
X_train, y_train = split_sequences(train_train_lstm2, n_steps)
print(X_train.shape, y_train.shape)
X_test, y_test = split_sequences(train_test_lstm2, n_steps)
print(X_test.shape, y_test.shape)

# The number of features
n_features = X_train.shape[2]

{% endhighlight %}

![Image_18](/agneev-blog/assets/img/img_5_18.png?raw=true){: width="200", height="100" }

After the preprocessing above, we are ready to define and run the LSTM model. I fit the model for 200 epochs, as I found this to give good results while taking a reasonable amount of time. If necessary the ideal number of epochs can be figured out as shown [here](https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/), but let's leave that for now.
{: style="text-align: justify"}

{% highlight python %}

# Define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(len(target_cols)))
model.compile(optimizer='adam', loss='mse')

# Fit model
model.fit(X_train, y_train, epochs=200, verbose=0)

# Demonstrate prediction
yhat = model.predict(X_test, verbose=0)

# Change X test array dimensions and concatenate with y test and y prediction arrays
X_test_new = X_test[:,0,:]
test_new = np.concatenate((X_test_new, y_test), axis=1)
test_new2 = np.concatenate((X_test_new, yhat), axis=1)

# Scale back the test arrays
test_new = scaler2.inverse_transform(test_new)
test_new2 = scaler2.inverse_transform(test_new2)

# Create target and prediction arrays containing only the target terms, to compare the predictions
y_targets = test_new[:, -len(target_cols):]
lstm_preds_1 = test_new2[:, -len(target_cols):]

{% endhighlight %}

<br>
Looking at the MAE values, we see that the LSTM model is more accurate for this waterbody than the RF and LGBM models.
{: style="text-align: justify"}

{% highlight python %}

mean_absolute_error(lstm_preds_1, y_targets, multioutput='raw_values')

mean_squared_error(lstm_preds_1, y_targets, multioutput='raw_values', squared=False)

{% endhighlight %}

![Image_19](/agneev-blog/assets/img/img_5_19.png?raw=true){: width="300", height="200" }

<br>
The SHAP analysis of the LSTM cannot be done by the TreeExplainer like for the tree-based models, and the DeepExplainer available for NNs gives errors like [this](https://github.com/slundberg/shap/issues/1490) for LSTMs, and so I used the [GradientExplainer](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html) instead. The difference in the SHAP summary plots for LSTM as compared to the earlier plots is obvious - the Depth_to_Groundwater terms are used directly in the LSTM for predictions, and hence high values of these variables give high values for the model output, and the other way around. This shows that the influence of a feature on the prediction depends not only on the variable but also on the type of model used.
{: style="text-align: justify"}

{% highlight python %}

explainer = shap.GradientExplainer(model, X_train)
shap_values1 = explainer.shap_values(X_test)
Xtests = np.hsplit(X_test, n_steps)
i = 0
for target in targets:
    shaps = np.hsplit(shap_values1[i], n_steps)
    shap.summary_plot(np.squeeze(shaps[n_steps-1]), np.squeeze(Xtests[n_steps-1])
                      ,feature_names = train_test_lstm.columns.tolist(), show=False)
    plt.title(f'{targets[i]}', fontsize=20)
    plt.show()
    i +=1

{% endhighlight %}

![Image_20](/agneev-blog/assets/img/img_5_20.png?raw=true){: width="600", height="300" }

![Image_21](/agneev-blog/assets/img/img_5_21.png?raw=true){: width="600", height="300" }

<br>
## Ensembling  <a id="Ensemble"></a>

We have seen that all the models gave quite good results on the Petrignano dataset. Let us now see how an ensemble of these models fares. First we need to reshape the different model prediction arrays to make these all the same shape. Then we take an average of the predicted values, and find the MAE.
{: style="text-align: justify"}

{% highlight python %}

rf_preds_1 = np.delete(rf_preds_1, np.s_[0:n_steps-1], axis=0)
lgbm_preds_1 = np.delete(lgbm_preds_1, np.s_[0:n_steps-1], axis=0)
lstm_preds_1 = lstm_preds_1.reshape(rf_preds_1.shape)
avg_preds_1 = (rf_preds_1 + lgbm_preds_1 + lstm_preds_1)/3

mean_absolute_error(avg_preds_1, y_targets, multioutput='raw_values')

{% endhighlight %}

![Image_22](/agneev-blog/assets/img/img_5_22.png?raw=true){: width="300", height="100" }

We see that the ensemble produces results that are slightly inferior to the best model here, the LSTM. Two comments can be made on this. Firstly, we have used a simple average of the predictions here. Using a [meta model to stack these predictions](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) will [almost certainly result](https://blogs.sas.com/content/subconsciousmusings/2017/05/18/stacked-ensemble-models-win-data-science-competitions/) in more accurate predictions. The downside is that stacked models are [significantly slower and more computationally expensive](https://towardsdatascience.com/a-practical-guide-to-stacking-using-scikit-learn-91e8d021863d). As I had already developed some very elaborate models that took a fair amount of time to run, I decided that a simple average would suffice here instead of adding further complications. The other point is that the benefit of ensembling will be more clearly brought out when I touch upon Aquifer Doganella.
{: style="text-align: justify"}

<br>
## Forecast period of 30 days  <a id="Thirty"></a>

I reran all the code for a forecast period of 30 days - everything remaining identical, except putting days_ahead = 30. The model ran smoothly and automatically, producing predictions that are slightly less accurate, understandable given that the prediction period is further away:
{: style="text-align: justify"}

![Image_23](/agneev-blog/assets/img/img_5_23.png?raw=true){: width="800", height="800" }

<br>
## Conclusion  <a id="Conc"></a>

This has been a very long post, since I had a lot to go over, and so I will leave it here for the moment. The next time, we will take a brief look at some of the interesting points in the modelling of the other water bodies to wind up this series! So long!
{: style="text-align: justify"}
