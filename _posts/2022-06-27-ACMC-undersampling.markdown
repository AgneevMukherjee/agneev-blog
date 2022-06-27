---
layout: post
title:  "9. ACMC - Undersampling"
date:   2022-06-26 12:00:00 +0100
category: ['Machine Learning', 'Computer Vision', 'Python']
tag: ['TensorFlow 2', 'Keras', 'Matplotlib', 'Seaborn', 'Pandas', 'Scikit-learn']
---

<span style="font-family:Helvetica; font-size:1.5em;">Categories:</span>
<div class="post-categories">
<p style="font-size:20px">
  {% if post %}
    {% assign categories = post.categories %}
  {% else %}
    {% assign categories = page.categories %}
  {% endif %}
  {% for category in categories %}
  <a href="{{site.baseurl}}/categories/#{{category|slugize}}">{{category}}</a>
  {% unless forloop.last %}&nbsp;{% endunless %}
  {% endfor %}
  </p>
</div>
<br/>

<span style="font-family:Helvetica; font-size:1.5em;">Tags:</span><br/>
<p style="font-size:18px">
{{page.tag | join: ', ' }}
</p>
<br/>

Hello! It's been a long time since my previous post, as I have had plenty to do, professionally and personally, since then. This also means that this post will be shorter than I had originally planned. [Last time](https://agneevmukherjee.github.io/agneev-blog/ACMC-simple-model/) I left off saying that we would be looking at different approaches for handling unbalanced datasets, but today we will only look at one such approach - the rest (hopefully) coming soon!
{: style="text-align: justify"}

So, as a brief reminder, we are dealing with the Archie Comics Multiclass (ACMC) dataset that I created and placed [here](https://www.kaggle.com/datasets/agneev/archie-comics-multi-class). We [have already seen](https://agneevmukherjee.github.io/agneev-blog/ACMC-simple-model/) how a simple model can get a decent validation accuracy of around 56%, at the cost of a) overfitting, and b) having a far higher true positive rate for the classes having more samples (i.e. the majority classes) in the unbalanced dataset. The overfitting we leave aside for the moment, but what about developing a model that is approximately equally accurate for every class? For this, we need to overcome the class imbalance, such as via...
{: style="text-align: justify"}

## Undersampling <a id="undersample"></a>

Since some of the classes having more samples than others is our issue, the first way of handling it would be to simply make the number of samples the same. This could be done either by reducing the number of samples considered in the majority classes (undersampling) or increasing the number of samples in the minority classes via duplication or synthetic sampling (oversampling). We will look at oversampling in a later post, but how do we go about undersampling? The simplest thing to do is to look at the minimum number of samples that are present in every class and just take that many images from each class. In other words, we reduce all the columns in the figure below to the red line showing the level of the minimum samples class (Svenson). Let's do this!
{: style="text-align: justify"}

![Image_1](/agneev-blog/assets/img/img_9_1.png?raw=true){: width="800", height="600" }

Now, I mentioned the 'simplest thing to do' above, but I found it surprisingly hard to actually carry it out in TensorFlow. After trying different methods, I finally decided that using the [flow_from_dataframe](https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c) function of Keras is the best way to go about things. It's not perfect - it is a function of Keras' [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator), which is deprecated. Still, this method allows us to quickly look at the effect of undersampling on the results without having to actually create new directories containing fewer images, etc., and so we'll stick with this for now. The full code is on Github [here](https://github.com/AgneevMukherjee/agneev-blog/blob/main/undersampling-acmc.ipynb) - let's look at the important parts.
{: style="text-align: justify"}

First, let's create and populate lists with the names of the classes, number of samples per class, and the image filenames:
{: style="text-align: justify"}

{% highlight python %}

samples_per_class = []
classes = []
file_names = []

directory=os.listdir('../input/archie-comics-multi-class/Multi-class/')
for each in directory:
    currentFolder = '../input/archie-comics-multi-class/Multi-class/' + each
    count = sum(len(files) for _, _, files in os.walk(currentFolder))
    samples_per_class.append(count)
    classes.append(each)

    for i, file in enumerate(os.listdir(currentFolder)):
            fullpath = currentFolder+ "/" + file
            file_names.append(fullpath)

min(samples_per_class)

{% endhighlight %}

The line at the end of the above code will show that the minimum number of samples in each class is 33. Now to go about undersampling. To do this, first let's create a list of 33 filenames per class. There are various ways of doing this, but the code below is what I have gone for.
{: style="text-align: justify"}

{% highlight python %}

small_ds = []

for each_class in classes:
    trial_list = []
    for name in file_names:
        if re.search(f'{each_class}', name):
            trial_list.append(name)
    small_ds.append(random.sample(trial_list, min(samples_per_class)))

{% endhighlight %}

This code deserves a bit of explanation. Essentially, after creating an empty small_ds list, we run a for-loop for each class populating this list. The for-loop itself works like this: consider any class, for instance 'Jughead'. We create a new empty trial_list, then, using a nested for-loop, scan through the file_names list we created earlier. Using a regular expression, we select the filenames containing 'Jughead'. These file names are added to trial_list. At the end of the nested for-loop, trial_list contains all the filenames containing 'Jughead' in them. Exiting the nested for-loop, we add 33 randomly selected 'Jughead' filenames from trial_list to the small_ds list. We then move to the next class in the outer loop, with a new empty trial_list being created, and so on. At the end, therefore, we get a list containing 33 randomly selected filenames for each of the 23 classes.
{: style="text-align: justify"}

Let us see whether we have created the small_ds properly. First, the length:
{: style="text-align: justify"}

{% highlight python %}

print(len(small_ds))
print(len(small_ds[0]))
print(len(small_ds)*len(small_ds[0]))

{% endhighlight %}

The results of the above print statements are 23, 33, and 759. In other words, we have created a 2D list with dimensions 23*33. Handling this list might be easier if it's 1D, so let's flatten it.
{: style="text-align: justify"}

{% highlight python %}

small_ds=list(np.concatenate(small_ds).flat)
print(len(small_ds))

{% endhighlight %}

The length is now 759, indicating we have successfully obtained a 1D list containing all the filenames. We can peek into this list and see:
{: style="text-align: justify"}

![Image_2](/agneev-blog/assets/img/img_9_2.png?raw=true){: width="600", height="400" }

The list, we see, contains the complete filenames selected randomly. Let us now make a Pandas dataframe out of this list.
{: style="text-align: justify"}

{% highlight python %}

files_df = pd.DataFrame(index=range(0, len(small_ds)),columns = ['Class'])

start = 0
end = min(samples_per_class)
for each_class in classes:
    files_df.iloc[start:end] = each_class
    start = end
    end = end + min(samples_per_class)

files_df['Class'] = files_df['Class'].astype('str')

{% endhighlight %}

Above, we first create an empty Pandas dataframe with an index between 0 and 759 and a solitary 'Class' column. We then fill up this 'Class' column using a for-loop - rows 0-32 containing the first class name, and so on.
{: style="text-align: justify"}

![Image_3](/agneev-blog/assets/img/img_9_3.png?raw=true){: width="400", height="200" }

The small_ds dataset is then added as a new column 'Files':

{% highlight python %}

files_df['Files'] = small_ds

{% endhighlight %}

We now see the files_df dataframe has the class names and the file names:

![Image_4](/agneev-blog/assets/img/img_9_4.png?raw=true){: width="400", height="200" }

Now, during my first run, I left things like this and moved on to modelling - this is an error! Try and figure out why, and we will return to the question a little later :grin:
{: style="text-align: justify"}

## Model run <a id="run"></a>

So the next step is to create training and validation generators with a 80:20 training-validation split. We can as usual apply a range of image augmentations to the training set here (as explained in [my earlier post](https://agneevmukherjee.github.io/agneev-blog/Roman-numerals-dataset-evaluation-part-2/#aug)).
{: style="text-align: justify"}

{% highlight python %}

datagen=ImageDataGenerator(validation_split=0.2)

batch_size = 8

train_generator=datagen.flow_from_dataframe(
dataframe=files_df,
directory=None,
x_col='Files',
y_col='Class',
subset="training",
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="categorical",
# rotation_range=30,
# width_shift_range=0.2,
# height_shift_range=0.2,
# brightness_range=(0.5,1.5),
# shear_range=0.2,
# zoom_range=0.2,
# channel_shift_range=30.0,
# fill_mode='nearest',
# horizontal_flip=True,
# vertical_flip=False,
target_size=(256,256))

valid_generator=datagen.flow_from_dataframe(
dataframe=files_df,
directory=None,
x_col='Files',
y_col='Class',
subset="validation",
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(256,256))

{% endhighlight %}

We can then run the model, with the [usual callbacks](https://agneevmukherjee.github.io/agneev-blog/Roman-numerals-dataset-evaluation-part-2/#stop). Initially, let us have a patience value of 15 epochs for the early stopping, and run it for 50 epochs. The image sizes I used were 256 x 256, the base value used last time. The first results look like this:
{: style="text-align: justify"}

![Image_5](/agneev-blog/assets/img/img_9_5.png?raw=true){: width="400", height="200" }
![Image_6](/agneev-blog/assets/img/img_9_6.png?raw=true){: width="400", height="200" }

Hmm...strange. The training accuracy and loss are fine, but the validation accuracy remains virtually at zero, while the validation loss actually rises relentlessly. Clearly something is wrong...
{: style="text-align: justify"}
.
{: style="text-align: justify"}
.
{: style="text-align: justify"}
.
{: style="text-align: justify"}

Remember I said earlier that creating the dataframe and moving directly to modelling was an error? Well, what this has resulted in is a complete mismatch between the training and validation sets. In Keras, "validation_split = 0.2" [leads to](https://github.com/keras-team/keras/issues/597) the bottom 20% of the input data being designated as the validation set while the upper 80% is the training set. This means that here, the training set consists of all the classes that appear in the upper portion of the files_df dataframe (Kleats, Midge, Dilton, etc.), while the bottom (Beazley, Hiram Lodge, Others...) becomes the validation set. No wonder then that the results are so poor, since the model is being trained on one set of classes and validated on another set!
{: style="text-align: justify"}

We therefore add a crucial line of code prior to creating the generators:
{% highlight python %}

files_df = files_df.sample(frac=1, random_state=1).reset_index()

{% endhighlight %}

The above line of code is explained well [here](https://datagy.io/pandas-shuffle-dataframe/) and so I won't go into the details, but in short, it shuffles the rows of the dataframe so that our training and validation sets now receive a mix of classes. If we see the last 10 rows of the dataframe, we now see:
{: style="text-align: justify"}

![Image_7](/agneev-blog/assets/img/img_9_7.png?raw=true){: width="400", height="200" }

<br>

That's better...the different classes are now randomly arranged. We see that the original indices have been added as a column, which we can drop if we choose. Anyway, if we run the model again, we now get:
{: style="text-align: justify"}

![Image_8](/agneev-blog/assets/img/img_9_8.png?raw=true){: width="350", height="200" }

We see that the validation accuracy and loss now look more familiar (and appropriate). Let us recap a sentence from the start of this post, however.
{: style="text-align: justify"}

> A simple model can get a decent validation accuracy of around 56%, at the cost of a) overfitting, and b) having a far higher true positive rate for the classes having more samples (i.e. the majority classes) in the unbalanced dataset.

In the simple model, we got the validation accuracy of 56% after 20 epochs. Here, after 50 epochs we managed around 34%. A cursory glance at the training curves shows that the overfitting problem has got worse. However, these things are not really surprising - we are using far fewer samples in this run, so understandably the accuracy figures suffer and overfitting occurs. The reason we experimented with undersampling in the first place was not to improve the validation accuracy or solve overfitting, but to make the results more equitable across the classes. Have we at least achieved that? Let us look at the [confusion matrix](https://agneevmukherjee.github.io/agneev-blog/ACMC-simple-model/#elephant). _(Note: the image looks much better on Firefox than Chrome, due to a well-known problem Chrome has with downscaling images. If needed, you can open the image in a new tab and zoom it to make it easier to read.)_
{: style="text-align: justify"}

![Image_9](/agneev-blog/assets/img/img_9_9.png?raw=true){: width="800", height="600" }

Viewed one way, the problem of the majority classes getting higher true positive rates has been resolved - the results for Archie, Jughead, etc. are pretty average now. On the other hand, several classes still perform much better or worse than average, perhaps because the characters are easier to identify, the validation images more closely resemble the training figures, or other factors. Our stated goal of achieving an even recognition of all the classes thus remains unaccomplished...
{: style="text-align: justify"}

## Conclusion <a id="conc"></a>
Undersampling is a straightforward method of dealing with unbalanced data. Unfortunately, it is not a very _effective_ method. Here, we discarded almost 90% of the original images in an attempt to balance the classes, but ended up with both a poor validation accuracy and an enduring disparity in the performance across classes. Fortunately, better options exist, one of which we will look at next time. So long!
{: style="text-align: justify"}

<div class="post-nav">
  <p>
    {% if page.previous.url %}
    <a href="{{ site.baseurl }}{{page.previous.url}}">&#8672;&nbsp;{{page.previous.title}}</a>
    {% endif %}
  </p>
  <p style = "text-align:right;">
    {% if page.next.url %}
    <a href="{{ site.baseurl }}{{page.next.url}}">{{page.next.title}}&nbsp;&#8674;</a>
    {% endif %}
  </p>
</div>
