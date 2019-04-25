Each directory D1 -> D7 contains data acquired during a different day by different people.

* D1 to D6 were acquired in different occasions in Ticino, Switzerland in 2017.
* D7 was acquired by MS and PhD students at Politecnico di Milano in Feb 2018.
* D8 was acquired by three middle school students at IDSIA in Mar 2018.
* D9 (very small) was acquired with a few bankers in 2018.

Each dataset contains one class per subdirectory:
* c0 - rock
* c1 - paper
* c2 - scissors

Images in the "testing" directory are sampled from datasets D1 -> D4 (and are not included in directories D1 -> D4) and are more or less evenly distributed among classes (31 images per class).

Therefore one can use the following splits.
* Train on D*, test on "testing" (easy, because testing images are from the same days/people of training images)
* Train on D1-D4 + "testing", test on remaining directories (hard, because testing images are from a completely different day that is not represented in the training data)
* Or any other combination that makes sense.

New datasets should be added to the repository as new directories.  For privacy reasons, avoid images with recognizable faces or any sort of identifying information.  All images should be preprocessed to be about 500x500 px, to limit filesize and speed up reading of images during live demonstrations.  You can use the resize_images python script.

