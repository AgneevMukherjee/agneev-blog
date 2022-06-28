---
layout: post
title:  "0. The Beginning!"
date:   2022-01-17 12:00:00 +0100
categories: General
---

<div class="post-nav">
  <p>
    {% if page.previous.url %}
    <big><b>
    <a href="{{ site.baseurl }}{{page.previous.url}}">&#8672;&nbsp;{{page.previous.title}}</a></b></big>
    {% endif %}
  </p>
  <p style = "text-align:right;">
    {% if page.next.url %}
    <big><b>
    <a href="{{ site.baseurl }}{{page.next.url}}">{{page.next.title}}&nbsp;&#8674;</a>
    </b></big>
    {% endif %}
  </p>
</div>

<br>

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

It all starts here! Over the course of the following months, I will be exploring different aspects related to data science/machine learning, perhaps with the occasional digression. Let's see how this journey goes!


<div class="post-nav">
  <p>
    {% if page.previous.url %}
    <big><b>
    <a href="{{ site.baseurl }}{{page.previous.url}}">&#8672;&nbsp;{{page.previous.title}}</a></b></big>
    {% endif %}
  </p>
  <p style = "text-align:right;">
    {% if page.next.url %}
    <big><b>
    <a href="{{ site.baseurl }}{{page.next.url}}">{{page.next.title}}&nbsp;&#8674;</a>
    </b></big>
    {% endif %}
  </p>
</div>
