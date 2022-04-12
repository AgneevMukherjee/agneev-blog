---
layout: post
title:  "7. Introducing the Archie Comics Multiclass dataset"
date:   2022-03-28 12:00:00 +0100
category: ['Machine Learning', 'Computer Vision', 'Python']
tag: ['Created dataset']
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

Hello again! Today marks the start of an exciting new series, where we look at image classification and other computer vision (CV)-related tasks using the Archie Comics Multi-class (ACMC) dataset that I created.
{: style="text-align: justify"}

Now, right at the start, let me address one issue that some of you may have - why comics? Why not something with more real world applicability? Well, firstly, because comics are inherently a sketched representation of the real world, almost any CV-related task that can be done on photo datasets can be done on comics datasets - image classification, object detection, image segmentation, image captioning, image reconstruction, you name it. Indeed, we shall look at several of these applications in later posts. The only real difference is that instead of say, training our model on photos of faces, cars, animals, etc., we will train on drawings of these instead. The second reason is that we can often obtain a large number of comics sketches much more easily than we could obtain photographs of a particular subject. In the ACMC dataset, for instance, we shall see that we have hundreds of sketches of certain characters, a number that may not be easy to obtain for any real life subjects. Thirdly, while comics may be representative of reality, they are often an _exaggerated_ version of reality, providing us with poses and expressions that are not common in real life. Depending on your point of view, this may or may not be a plus, but I see it as increasing the diversity of the dataset. Fourth, and probably the main reason I chose to go down this path - it's fun! Look, doing machine learning (ML) stuff is often hard and frustrating, and so I preferred to choose a subject that is at least interesting to me, rather than some dataset I couldn't care less about!
{: style="text-align: justify"}

OK, so then comes the next question - why Archie comics in particular? Once again, we have a list of reasons...starting with the fact that I have been reading them for almost three decades now, and so both have an extensive amount of material, and an intimate knowledge of the characters. The second reason would be that they have been published for over 80 years now. Why does that matter? Well, aside from the abundance of available material, this leads to a point that may be better appreciated by those familiar with the comics than those who are not. You see, dozens of artists have drawn these characters over the decades, bringing their own drawing style to the table. At the same time, dressing styles and other aspects of daily life have also changed tremendously over this period. Despite this, the artists are required to maintain a continuity with past representations of the characters, so that readers can read a 2020 Archie story as easily as they can a 1950 story (and stories from different decades often appear together in the same issue). In other words, the characters have to look similar enough for human readers to be able to readily recognise them. This, then, is a great CV challenge - can the model learn the character feature representations well enough to tell that, though they look _somewhat_ different, this sketch from 1962 and this from 2008 both show Reggie?
{: style="text-align: justify"}

Thirdly, they have a large cast of characters - over two dozen recurring characters, easily. This makes it especially interesting for classification-type tasks. Fourthly, the frequency of appearance of these characters is wildly different, with those of the lead characters often orders of magnitude higher than some of the niche characters. This makes the dataset very imbalanced...and even more intriguing!
{: style="text-align: justify"}

So let's look at the dataset, which can be found [here](https://www.kaggle.com/datasets/agneev/archie-comics-multi-class), in a little more detail. Before that, an obvious disclaimer - all rights to the images belong to Archie Comic Publications Inc., with the dataset only being used by me for educational purposes. I have created the dataset from clippings from various Archie comic books and newspaper strips, with some minor editing occasionally done to remove the lettering in dialogue boxes, etc.
{: style="text-align: justify"}

## Multi-class versus multi-label <a id="mc-ml"></a>
 Before anything else, I should mention that I have actually created _two_ datasets - one multi-class, and one multi-label. We will deal with the multi-label dataset later, but since people sometimes get confused about the difference between multi-class and multi-label classification, let me explain this here briefly. Multi-class is when every image has a single label, with the label being one of a number of possible classes. Multi-label is when every image has or can have multiple labels.
{: style="text-align: justify"}

An example of multi-class image classification is this:

 ![Image_1](/agneev-blog/assets/img/img_7_1.png?raw=true){: width="150", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 ![Image_2](/agneev-blog/assets/img/img_7_2.png?raw=true){: width="200", height="100" }
<p> &emsp;  &emsp;   &emsp; Cat &emsp;  &emsp;  &emsp;  &emsp;  &emsp;  &emsp;  &emsp;  &emsp;  &emsp; Dog</p>
 <p style="color:grey;font-size:80%;">
© 2022 Agneev Mukherjee
</p>
<br>
Simple, right? On the other hand, the images below, with the labels given below them, may be used for multi-*label* classification:

![Image_3](/agneev-blog/assets/img/img_7_3.jpg?raw=true){: width="400", height="300" }
<br>
[Grass; Sand; Sea; Boats; Sunny]
<p style="color:grey;font-size:80%;">
© 2022 Agneev Mukherjee
</p>
<br><br>
![Image_4](/agneev-blog/assets/img/img_7_4.jpg?raw=true){: width="400", height="200"}
<br>
[Cars; Bicycle; Buildings; Trees; Lamp post; Street; Grass; Shrubs; Cloudy]
<p style="color:grey;font-size:80%;">
© 2022 Agneev Mukherjee
</p>
<br>

Of course, the exact labels for the figures will depend on the application, but you get the drift. So the dataset we are dealing with now is multi-class, that is, each picture only has a single label - the name of a particular character - attached to it. Later we will deal with the multi-label dataset, with several characters in each image.
{: style="text-align: justify"}

## Brief look at the characters <a id="brief"></a>

As I said earlier, Archie comics has a venerable history stretching back over 80 years. While nowhere near as popular (or ubiquitous) as in its heyday, it continues to have legions of fans, with new 'properties' like the animated series [_Archie's Weird Mysteries_](https://en.wikipedia.org/wiki/Archie%27s_Weird_Mysteries), the zombie comics title [_Afterlife with Archie_](https://en.wikipedia.org/wiki/Afterlife_with_Archie), and the TV series [_Riverdale_](https://en.wikipedia.org/wiki/Riverdale_(2017_TV_series)) coming out every now and then. There are also different versions of the characters like [_Little Archie_](https://en.wikipedia.org/wiki/Little_Archie), [_The New Archies_](https://en.wikipedia.org/wiki/The_New_Archies), the ['new look' series](https://en.wikipedia.org/wiki/Bad_Boy_Trouble), etc.  The ACMC, however, deals with the _classic_ version of the Archie comics characters - as we shall see, there is more than enough varied content in this to satiate us.
{: style="text-align: justify"}

I was originally planning to provide a fairly detailed description of each character in the dataset, but then realised that this would be superfluous and irrelevant to the task at hand. I therefore redirect you to [Wikipedia](https://en.wikipedia.org/wiki/List_of_Archie_Comics_characters) if you would like to know more about them. Here, let's just see a couple of pix of each character, one 'old' and one 'new'.
{: style="text-align: justify"}

Archie Andrews (_An archetypal average American teenager_):

![Image_5](/agneev-blog/assets/img/img_7_5.png?raw=true){: width="150", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_6](/agneev-blog/assets/img/img_7_6.png?raw=true){: width="180", height="100" }

<br>
Jughead Jones (_Archie's best friend, usually wears a beanie, smart but lazy, girl hater, has an insatiable appetite_):

![Image_7](/agneev-blog/assets/img/img_7_7.png?raw=true){: width="150", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_8](/agneev-blog/assets/img/img_7_8.png?raw=true){: width="170", height="90" }

<br>
Betty Cooper (_Sweet girl, good at studies and sports, has crush on Archie_):

![Image_9](/agneev-blog/assets/img/img_7_9.png?raw=true){: width="200", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_10](/agneev-blog/assets/img/img_7_10.png?raw=true){: width="150", height="80" }

<br>
Veronica Lodge (_Rich, spoilt girl, Archie's crush_):

![Image_11](/agneev-blog/assets/img/img_7_11.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_12](/agneev-blog/assets/img/img_7_12.png?raw=true){: width="160", height="100" }

<br>
Reggie Mantle (_Archie's main rival_):

![Image_13](/agneev-blog/assets/img/img_7_13.png?raw=true){: width="150", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_14](/agneev-blog/assets/img/img_7_14.png?raw=true){: width="170", height="100" }

<br>
The above are, in my opinion, the 5 most important characters in the 'Archies Universe' - and, in fact, the members of the band ['The Archies'](https://en.wikipedia.org/wiki/The_Archies)! The 'Riverdale gang', on the other hand, has several other members, among whom the most notable are:
{: style="text-align: justify"}

Dilton Doiley (_Stereotypical teenage geeky genius_):

![Image_15](/agneev-blog/assets/img/img_7_15.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_16](/agneev-blog/assets/img/img_7_16.png?raw=true){: width="180", height="100" }

<br>
Moose Mason (_Stereotypical jock, with near-superhuman strength and a meagre intellect_):

![Image_17](/agneev-blog/assets/img/img_7_17.png?raw=true){: width="150", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_18](/agneev-blog/assets/img/img_7_18.png?raw=true){: width="120", height="60" }

<br>
Midge Klump (_Moose's girlfriend_):

![Image_19](/agneev-blog/assets/img/img_7_19.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_20](/agneev-blog/assets/img/img_7_20.png?raw=true){: width="120", height="60" }

<br>
Ethel Muggs (_Chases Jughead_):

![Image_21](/agneev-blog/assets/img/img_7_21.png?raw=true){: width="180", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_22](/agneev-blog/assets/img/img_7_22.png?raw=true){: width="100", height="60" }

<br>
Chuck Clayton (_Talented athlete and cartoonist_):

![Image_23](/agneev-blog/assets/img/img_7_23.png?raw=true){: width="180", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_24](/agneev-blog/assets/img/img_7_24.png?raw=true){: width="150", height="80" }

<br>
Nancy Woods (_Chuck's girlfriend_):

![Image_25](/agneev-blog/assets/img/img_7_25.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_26](/agneev-blog/assets/img/img_7_26.png?raw=true){: width="100", height="50" }

<br>
The Riverdale gang studies at Riverdale high, whose most important staff members are:

Waldo Weatherbee (_The school principal_):

![Image_27](/agneev-blog/assets/img/img_7_27.png?raw=true){: width="200", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_28](/agneev-blog/assets/img/img_7_28.png?raw=true){: width="150", height="80" }

<br>
Geraldine Grundy (_Usually an English teacher, although often also shown teaching Maths, History and other subjects_):

![Image_29](/agneev-blog/assets/img/img_7_29.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_30](/agneev-blog/assets/img/img_7_30.png?raw=true){: width="150", height="80" }

<br>
Mr. Flutesnoot (_Usually a science, especially chemistry, teacher_):

![Image_31](/agneev-blog/assets/img/img_7_31.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_32](/agneev-blog/assets/img/img_7_32.png?raw=true){: width="140", height="70" }

<br>
Coach Kleats (_Head physical education teacher_):

![Image_33](/agneev-blog/assets/img/img_7_33.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_34](/agneev-blog/assets/img/img_7_34.png?raw=true){: width="120", height="60" }

<br>
Coach Clayton (_Chuck's father, physical education and history teacher_):

![Image_35](/agneev-blog/assets/img/img_7_35.png?raw=true){: width="150", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_36](/agneev-blog/assets/img/img_7_36.png?raw=true){: width="120", height="60" }

<br>
Mr. Svenson (_School janitor_):

![Image_37](/agneev-blog/assets/img/img_7_37.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_38](/agneev-blog/assets/img/img_7_38.png?raw=true){: width="150", height="80" }

<br>
Ms. Beazley (_School cafeteria cook_):

![Image_39](/agneev-blog/assets/img/img_7_39.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_40](/agneev-blog/assets/img/img_7_40.png?raw=true){: width="150", height="80" }

<br>
Of the parents of the Riverdale gang, two characters make appearances far more frequently than others:

Hiram Lodge (_Veronica's father_):

![Image_41](/agneev-blog/assets/img/img_7_41.png?raw=true){: width="120", height="60" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_42](/agneev-blog/assets/img/img_7_42.png?raw=true){: width="150", height="80" }

<br>
Fred Andrews (_Archie's father_):

![Image_43](/agneev-blog/assets/img/img_7_43.png?raw=true){: width="150", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_44](/agneev-blog/assets/img/img_7_44.png?raw=true){: width="140", height="70" }

<br>
Being as rich as they are, it is no surprise that the Lodges have a butler, Smithers, who makes semi-regular appearances:

![Image_45](/agneev-blog/assets/img/img_7_45.png?raw=true){: width="150", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_46](/agneev-blog/assets/img/img_7_46.png?raw=true){: width="140", height="70" }

<br>
And what is the gang's favourite hangout? Pop Tate's Chok'lit Shoppe, of course. Here's Pop:

![Image_47](/agneev-blog/assets/img/img_7_47.png?raw=true){: width="160", height="80" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_48](/agneev-blog/assets/img/img_7_48.png?raw=true){: width="130", height="70" }

<br>
So that was a round-up of all the major characters in the ACMC dataset. However, the dataset actually has one more class - 'Others'. This, as you can guess, is a medley of images of random characters, and the aim is for a model to put any images that it cannot classify as a member of any of the other classes into this category. Let's have a look at some of the images under this heading:
{: style="text-align: justify"}

![Image_49](/agneev-blog/assets/img/img_7_49.png?raw=true){: width="130", height="100" }
![Image_50](/agneev-blog/assets/img/img_7_50.png?raw=true){: width="130", height="100" }
![Image_51](/agneev-blog/assets/img/img_7_51.png?raw=true){: width="130", height="100" }
![Image_52](/agneev-blog/assets/img/img_7_52.png?raw=true){: width="130", height="100" }
![Image_53](/agneev-blog/assets/img/img_7_53.png?raw=true){: width="120", height="100" }

<br>
![Image_54](/agneev-blog/assets/img/img_7_54.png?raw=true){: width="110", height="100" }
![Image_55](/agneev-blog/assets/img/img_7_55.png?raw=true){: width="140", height="120" }
![Image_56](/agneev-blog/assets/img/img_7_56.png?raw=true){: width="140", height="120" }
![Image_57](/agneev-blog/assets/img/img_7_57.png?raw=true){: width="140", height="120" }
![Image_58](/agneev-blog/assets/img/img_7_58.png?raw=true){: width="140", height="120" }

<br>
![Image_59](/agneev-blog/assets/img/img_7_59.png?raw=true){: width="160", height="120" }
![Image_60](/agneev-blog/assets/img/img_7_60.png?raw=true){: width="90", height="60" }
![Image_61](/agneev-blog/assets/img/img_7_61.png?raw=true){: width="160", height="120" }
![Image_62](/agneev-blog/assets/img/img_7_62.png?raw=true){: width="110", height="80" }
![Image_63](/agneev-blog/assets/img/img_7_63.png?raw=true){: width="140", height="120" }

<br>
Archie comics fans will recognise several familiar faces in that gallery: Jughead's father, Gaston, Archie's mother, Cheryl Blossom, Betty's father, Jellybean, and Ms. Haggly. The other images are of non-recurring characters.
{: style="text-align: justify"}

## Conclusion <a id="conc"></a>

Looking at the pictures above serves to highlight some of the challenges of working with this dataset. I used a range of materials to make the dataset, collected over a long period, and hence the images vary widely in size and image quality. How did I decide whether an image belongs in the dataset or not? The primary criterion was that a 'human expert', in this case an Archie comics fan, should be able to look at the image in question and identify it without any further cues. In some cases, this is difficult - without prior knowledge, for example, it is hard to categorise the Midge images above as belonging to the same person. Still, this is better than if I had put Jughead's mother as a category - believe it or not, both the images below are of her. These images actually appear in the 'Others' category, meaning that their belonging to the same character is moot.
{: style="text-align: justify"}

![Image_64](/agneev-blog/assets/img/img_7_64.png?raw=true){: width="180", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_65](/agneev-blog/assets/img/img_7_65.png?raw=true){: width="120", height="60" }

<br>
Another reason for the dataset being challenging - well, a look at some Jughead images below should suffice...
{: style="text-align: justify"}

<br>
![Image_66](/agneev-blog/assets/img/img_7_66.png?raw=true){: width="100", height="100" }&nbsp;&nbsp;
![Image_67](/agneev-blog/assets/img/img_7_67.png?raw=true){: width="100", height="100" }&nbsp;&nbsp;
![Image_68](/agneev-blog/assets/img/img_7_68.png?raw=true){: width="100", height="100" }&nbsp;&nbsp;
![Image_69](/agneev-blog/assets/img/img_7_69.png?raw=true){: width="100", height="100" }&nbsp;&nbsp;
![Image_70](/agneev-blog/assets/img/img_7_70.png?raw=true){: width="100", height="100" }&nbsp;&nbsp;
![Image_71](/agneev-blog/assets/img/img_7_71.png?raw=true){: width="100", height="100" }

<br>
The images above can all easily be identified by any Jughead fan, but for a ML model, it might not be as straightforward.
{: style="text-align: justify"}

And finally, as I said at the start, the dataset is also very imbalanced, which introduces its own challenges - and provides the opportunity to test some novel techniques. We shall look at this and other aspects in detail next time, so goodbye for now!
{: style="text-align: justify"}
