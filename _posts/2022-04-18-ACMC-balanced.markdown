---
layout: post
title:  "9. Archie Comics Multiclass dataset-balanced"
date:   2022-04-18 12:00:00 +0100
category: ['Machine Learning', 'Computer Vision', 'Python']
tag: ['TensorFlow 2', 'Keras', 'Matplotlib', 'Seaborn', 'PIL', 'Pandas', 'Scikit-learn']
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

Welcome back. We left off [last time] by posing the question - while a conventional ResNet treatment worked well enough in terms of classifying the major classes (Archie, Betty, Jughead, etc.) of the [Archie Comics Multiclass Dataset](https://www.kaggle.com/datasets/agneev/archie-comics-multi-class) (ACMC), this approach performed very poorly on the classes with few samples, which is problematic if a classifier that is even-handed in recognising the different classes is what we want. In this post, we will look at some of the approaches we can take to rectify this.
{: style="text-align: justify"}

## A look at the images and their distribution <a id="look"></a>

{% highlight python %}


{% endhighlight %}


![Image_5](/agneev-blog/assets/img/img_8_5.png?raw=true){: width="400", height="200" }

<p style="color:grey;font-size:100%;">
 Image 5: Accuracy and loss of ResNet50 model with Imagenet weights
</p>
