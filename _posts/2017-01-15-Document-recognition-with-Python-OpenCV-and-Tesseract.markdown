---
layout: post
title:  "Document recognition with Python, OpenCV and Tesseract"
date:   2017-01-15 15:49:38 +0200
tags: opencv python tesseract ocr
comments: true
---

Recently I’ve conducted my own little experiment with the document recognition technology: I’ve successfully went from an image to the recognized editable text.
On the way I heavily relied on the two following articles:

1. [Build a Kick-Ass Mobile Document Scanner in Just 5 Minutes](http://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/)
2. [Finding blocks of text in an image using Python, OpenCV and numpy](http://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html)

However, I’ve added something myself, and that’s what I want to write about: the ways to improve upon the given articles to achieve the goal of recognizing plain text from photos at arbitrary angles and illumination.

### Skew removal (first article)

* If the edges in your image aren’t recognized properly, it makes sense to move interval in Canny function to the left. For example, I’ve settled on [40, 150], instead of [75, 200] proposed in the article.
* To be able to recognize documents with less straight than rectangular shapes (for example, with curved corners), you should apply blurring to the image with detected edges. I found that it improves approximation quality even in some cases of “good” document shapes, and decreases in none.

Resulting image at this point can be fed to the Tesseract engine, but to get better results out of it, we can conduct two more steps:

### Noise removal
I’ve experimented with different combinations of Gaussian blur, median blur, erosion and dilation (these last two can sound scary, but they aren’t. See [Eroding and Dilating](http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html)).

* Gaussian blur corrupts text more than median blur
* I found that using only dilation yields better average results than any other combination of mentioned techniques. 
* In addition to removing noise, dilation makes text clearer (by making white spaces inside of letter such as ‘a’ or ‘e’ bigger)

### Cropping (second article)
* To achieve quicker processing, the image at this step can be resized not to the height of 2048, but of 600 (I wouldn’t recommend anything less than 500). I haven’t observed any negative effects after introducing this change.
* Two rank filters can be replaced by a single median blur
* Shape of the dilation kernel doesn’t affect the result

Now it’s time to feed the resulting image to the Tesseract. It can be done quite simply with:

{% highlight python %}
pytesseract.image_to_string(Image.open('scan_res.jpg'))
{% endhighlight %}

Additionally, I want to mention the problem of measuring accuracy of recognition. Having found nothing on Google in the first five minutes of the search, I’ve coded my own metric, which turned out to be not that good. That forced me to google some more, and so I’ve found a paper that discusses and compares different metrics (on their own dataset, which is quite different from mine: Arabic handwriting isn’t exactly English words in print). It gave me the names of the metrics however, which was good enough. The paper can be found [here](http://www.ijcsi.org/papers/IJCSI-11-3-1-18-26.pdf).

I’ve settled on Jaro-Winkler distance, which gives sufficiently accurate results. Also it’s implemented in jellyfish package, so it’s quite easy to use.
I hope this article will prove itself useful. 
For reference, my final code can be found [here](https://github.com/AwesomeLemon/document-recognition)