# pset2 for CS 378 Intro to Speech and Audio Processing

1. Report the final accuracy achieved by your model : 58% 
2. What are the 3 phoneme classes that your model predicts with the highest accuracy? s, b, y (it also classifies silence well)
3. What about the 3 classes that have the lowest accuracy? vcl, ah, ih
4. For phoneme segments that have the ground-truth label ’sh’, what other phoneme class are they most commonly mis-classified as? Does this make sense? Why or why not? s, and it makes sense, as the 'sh' sound and 's' sound are very similar
5. Repeat the previous question for the ’p’, ’m’, ’r’, and ’ae’ phoneme classes.

p: b, and this makes sense as puh and buh are very similar, pave, brave, etc sound similar at the start depending on how you say them

m: ng, this was a little less sense, but if you are in the middle of a word a m sound could be similar

r: er, this makes a lot of sense, as it it just a different version

ae: eh, these are very similar, just based on length
