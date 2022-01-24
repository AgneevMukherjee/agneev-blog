---
layout: post
title:  "1. Extensive Roman MNIST dataset – the preparation"
date:   2022-01-24 12:00:00 +0100
category: data science
categories:
---
## A novel data science competition...

If you have ever been involved in data science (DS) or machine learning (ML), even briefly or tangentially, you will be well aware of data science competitions. These competitions are an invaluable learning experience for beginning data scientists, while even seasoned professionals often continue participating in these to keep their skills fresh, learn new tricks, interact with fellow competitors, or simply for the fun of it. Oh, and many competitions offer a fair amount of cash to further sweeten the deal...:wink:

[Kaggle](https://www.kaggle.com/competitions) is the largest and most famous DS/ML competition platform, although there are tons of others - [DrivenData](https://www.drivendata.org/competitions/), [AIcrowd](https://www.aicrowd.com/challenges), [Xeek](https://xeek.ai/challenges), [Zindi](https://zindi.africa/competitions), etc. Most of the competitions held by these platforms follow a similar pattern - the competitors are provided with a dataset and asked to make a model that provides the most accurate predictions for the target variable(s). In other words, the data is held constant, and the models tuned to fit the data.

In June 2021, [Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng) announced a [Data-Centric AI competition](https://https-deeplearning-ai.github.io/data-centric-comp/), which turned the problem on its head. Here, the model (a modified ResNet50, which is a convolutional neural network - more on these in a later post...) was kept fixed, and the competitors were asked to modify the image data provided in any way they saw fit, subject to a maximum of 10,000 images. This was an interesting challenge, and in line with Dr. Ng's philosophy that ML technology like neural networks have progressed far enough that major future advances in their application must come not via finetuning their architectures but through improvements in the data fed to these models, an area that has been neglected so far.

So what was the challenge itself? It was to enable the model provided to recognise hand-written Roman numerals. In other words, the competitors had to create a Roman numerals version of the famous [Modified National Institute of Standards and Technology (MNIST)](https://en.wikipedia.org/wiki/MNIST_database) dataset. The winners were selected on two tracks - the best overall leaderboard score, and the most innovative approach, as decided by a jury. Unfortunately, my entries did not finish in the top three in either category, but no matter - it was a great learning experience! This blog post will focus on my dataset preparation, while next week we will look at training the model and evaluating the dataset.

## The data provided

As mentioned above, the competition organisers provided some data to get started. This data was grouped into two folders - train and val - with each folder have ten sub-folders - i to x. As you can guess, each subfolder contained images of handwritten Roman numerals, from 1 to 10. The number of images in each folder varied - for the training folders, from 157 to 281, and for the validation folders, from 77 to 84. As part of the challenge, one was free to move images from the training to validation folders, or the other way around, as desired, augment or curtail the number of images, or do anything else one saw fit, as long as the _total_ number of images remained below 10,000. A quick glance at the data, though, made clear what the very first step ought to be...

## Removing bad data

The first thing one could see while looking at the images provided was that many of them were...strange. Have a look:

![Image_1](/agneev-blog/assets/img/img_1_1.png?raw=true){: width="150", height="125" }
![Image_2](/agneev-blog/assets/img/img_1_2.png?raw=true){: width="150", height="125" }
![Image_3](/agneev-blog/assets/img/img_1_3.png?raw=true){: width="150", height="125" }
![Image_4](/agneev-blog/assets/img/img_1_4.png?raw=true){: width="150", height="125" }
![Image_5](/agneev-blog/assets/img/img_1_5.png?raw=true){: width="150", height="125" }
![Image_6](/agneev-blog/assets/img/img_1_6.png?raw=true){: width="150", height="125" }
![Image_7](/agneev-blog/assets/img/img_1_7.png?raw=true){: width="150", height="125" }
![Image_8](/agneev-blog/assets/img/img_1_8.png?raw=true){: width="150", height="125" }

While deciding on which image to remove, one must be careful not to make the images left behind _too_ clean - after all, the model must learn to recognise images even if they are not squeaky clean and perfectly written. A good thumb rule, therefore, is to remove images that you yourself are unable to recognise, and keep the rest. The pix shown above are clearly undesirable, and so these, and similar images, were removed. There were also several instances of images being in the wrong folder (e.g. 5 or 10 in the folder for 2), and these were put in the right place.

## Gathering own data

Eliminating all the bad images left something like 2500 images in all, well below the max limit of 10,000. In general, deep learning systems tend to perform better with more data, which meant that gathering more images snapped in different settings would be a good way to make the dataset more diverse and robust. My way of doing this was relatively straightforward – I clicked pictures of numbers I wrote myself in a variety of styles and conditions, and asked as many relatives and friends as I could, without their thinking I was crazy, to send me their handwritten Roman numerals. Chopping the images into the individual numbers was a surprisingly time-consuming and laborious task, and one which made me appreciate afresh the challenges in gathering good quality data. Nevertheless, I was able to gather 300+ images for each number. At the time, I didn’t know whether these resembled the test data or not (spoiler: they didn't), but I anyway attempted to gather the most diverse set of images possible. Some samples are given below.

![Image_9](/agneev-blog/assets/img/img_1_9.png?raw=true){: width="150", height="125" }
![Image_10](/agneev-blog/assets/img/img_1_10.png?raw=true){: width="150", height="125" }
![Image_11](/agneev-blog/assets/img/img_1_11.png?raw=true){: width="150", height="125" }
![Image_12](/agneev-blog/assets/img/img_1_12.png?raw=true){: width="150", height="125" }
![Image_13](/agneev-blog/assets/img/img_1_13.png?raw=true){: width="150", height="125" }
![Image_14](/agneev-blog/assets/img/img_1_14.png?raw=true){: width="150", height="125" }
![Image_15](/agneev-blog/assets/img/img_1_15.png?raw=true){: width="150", height="125" }
![Image_16](/agneev-blog/assets/img/img_1_16.png?raw=true){: width="150", height="125" }

## Data quantisation

The organisers provided a script for optionally processing the added images so as to make them more similar to data already provided. The script below, which uses the OpenCV library, loads the images in grayscale mode and converts all the pixels that aren't very dark (brightness of 43 or less) to white. The results can be seen below, with an original image to the left, and the quantised image to the right.

{% highlight ruby %}
def convert_images(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    input_files = glob(os.path.join(input_folder, "*.png"))
    for f in input_files:
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # quantize
        image = (image // 43) * 43
        image[image > 43] = 255
        cv2.imwrite(os.path.join(output_folder, os.path.basename(f)), image)
{% endhighlight %}

![Image_17](/agneev-blog/assets/img/img_1_17.png?raw=true){: width="150", height="125" }&nbsp;&nbsp;&nbsp;&nbsp;
![Image_18](/agneev-blog/assets/img/img_1_18.png?raw=true){: width="150", height="125" }

Although this was optional, I chose to undertake this conversion anyway. Since the data provided was in black and white, I felt the test data was unlikely to be in colour, and so would probably resemble the processed images more than the original colour versions (this turned out to be true, btw).

## Data manipulation for augmentation

Now, even after gathering my own data, I ended up with less than 6000 images. How to boost the numbers further? One method is via manipulating the existing images. An easy way to do this is flipping the images using OpenCV's flip method. The small versions of 1, 2, 3 and 10 can be flipped horizontally, while their capital versions can be flipped either horizontally and vertically. For 5, only the horizontal flip is meaningful, while for 9, only the capital 9 can be flipped vertically. Examples:

![Image_19](/agneev-blog/assets/img/img_1_19.png?raw=true){: width="125", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;
![Image_20](/agneev-blog/assets/img/img_1_20.png?raw=true){: width="125", height="100" }

![Image_21](/agneev-blog/assets/img/img_1_21.png?raw=true){: width="125", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;
![Image_22](/agneev-blog/assets/img/img_1_22.png?raw=true){: width="125", height="100" }

![Image_23](/agneev-blog/assets/img/img_1_23.png?raw=true){: width="125", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;
![Image_24](/agneev-blog/assets/img/img_1_24.png?raw=true){: width="125", height="100" }

For the numbers 4 and 6, I flipped the numbers horizontally, and put the results in the other number's folder. Here's what I mean...the images to the left below are the original, and to the right the flipped versions.

![Image_25](/agneev-blog/assets/img/img_1_25.png?raw=true){: width="125", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;
![Image_26](/agneev-blog/assets/img/img_1_26.png?raw=true){: width="125", height="100" }

![Image_27](/agneev-blog/assets/img/img_1_27.png?raw=true){: width="125", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;
![Image_28](/agneev-blog/assets/img/img_1_28.png?raw=true){: width="125", height="100" }

Obviously some flipped images had to be eliminated because the flipped version didn't quite look right (see the 4 below), while in some others, some manual changes were necessary (the dot of the flipped 9 moved to the top):

![Image_29](/agneev-blog/assets/img/img_1_29.png?raw=true){: width="125", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Image_30](/agneev-blog/assets/img/img_1_30.png?raw=true){: width="125", height="100" }&nbsp;&nbsp;&nbsp;&nbsp;
![Image_31](/agneev-blog/assets/img/img_1_31.png?raw=true){: width="125", height="100" }

Unfortunately, no sensible flips are possible for the numbers 7 and 8, and so these therefore need to be augmented in a different way. I manually added an ‘i’ to vii’s and removed an ‘i’ from viii’s, as shown below (original to the right in each pair). While effective, this method was laborious and time-consuming, taking me over half an hour to generate a hundred images.
![Image_32](/agneev-blog/assets/img/img_1_32.png?raw=true){: width="150", height="125" }
![Image_33](/agneev-blog/assets/img/img_1_33.png?raw=true){: width="150", height="125" }
![Image_34](/agneev-blog/assets/img/img_1_34.png?raw=true){: width="150", height="125" }
![Image_35](/agneev-blog/assets/img/img_1_35.png?raw=true){: width="150", height="125" }

Overall, while the flipping and manual modification methods worked, they did not add all that much diversity to the dataset, and the manual manipulation in particular was very time consuming. I therefore used another method to generate some more images.
