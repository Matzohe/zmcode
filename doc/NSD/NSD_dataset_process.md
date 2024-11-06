# NSD数据处理流程

## 图像信息采集 extract image list
对于NSD数据集，被试看到的图像的信息保存格式为
subject%d_rep%01d，含义为第几位被试第几次看到的图像，比如subject6_rep1

NSD保存被试看到的图像信息为一个pkl文件，在这个pkl文件中，总共拥有73000条信息，每条信息代表着一张图像。
对于每一条记录，存放着如下的一些信息
1、cocoId与存放的coco文件夹位置（val2017 or train2017）
2、cropBox与loss（？不是很清楚这一块的作用是什么）
3、nsdId：图像在NSD数据集中对应的ID，从0-72999
4、flagged，BOLD5000（？不是很清楚这一块的作用是什么）
5、shared1000:是否是分享的1000张图像
6、subject%d_rep%01d，（应该是第几位被试第几次看到这张图像的时间，从1-30000，如果被试没有见过这张图，就是0）在提取完成后会进行一个减一的操作，便于后续提取相应的信息。

将每位被试的看到图像的激活和对应的image index取出（由于列表是有顺序的，因此图像激活的矩阵（10000 * 3）和image index的列表（1 * 10000）是相互对应的，两者都是根据subject%d_rep%01d是否为0取的，将所有不为零的部分取出）

## 皮层信号处理，extract cortical voxal

皮层信号处理分为两部分，第一部分是提取对应脑区cortex的脑响应的mask，第二部分是将对应脑区响应取出，然后计算zscore

### 皮层mask提取
对应脑区的mask的保存路径为
/data/guoyuan/nsd/nsddata/ppdata/subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz
当使用了nsd_data.get_fdata()之后，获得了一个mask，这个mask是用-1代表不行的区域。整体的脑区的大小为（79 * 97 * 78 = 597714）个体素
在nsdgeneral这个mask里面，mask对应的shape就是目标视觉皮层脑区的shape，而有效的部分它的标记是2，而在selective roi中，选出的体素数是大于感兴趣的脑区的体素数（这里需要确认一下，在roi的整个列表中，那些标签代表的含义是什么）

### 皮层响应zscore计算
原始数据保存在
/data/guoyuan/nsd/nsddata_betas/ppdata/beta_session%02d.nii.gz
从NSD论文可知，对于每一个被试，拥有四十个session，每个session会看750张图像，而在提取过程中，对session循环，提取出每位被试四十个session的所有信息。

对于每位被试的每个session，这个beta_session里存的内容是（79*97*78*750）个脑区响应，也就是说，beta session已经对fMRI响应做了一次处理，得到的是一个平均值

在计算过程中，将每一次的脑区响应取一个zscore，然后保存，这就是NSD的脑响应处理过程。


## CLIP2Brain相同图像不同rep处理方法 compute ev
在clip2brain中，对于相同图像的数据，作者将脑区响应取了简单的平均，用来代表被试在这张图像下的脑区响应