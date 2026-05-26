Full length article

# Transformer-based 3D range-gated imaging method with multiple depth priors

Xiaoquan Liu ${ \mathrm { a } } , { \mathrm { b } } _ { \oplus }$ , Di Zhang a,b , Jinming Gao a,b , Xinwei Wang c,d,e,\* a Hubei Key Laboratory of Intelligent Robot, Wuhan Institute of Technology, 430205 Wuhan, China b School of Computer Science and Engineering, Wuhan Institute of Technology, 430205 Wuhan, China c Optoelectronics System Laboratory, the Institute of Semiconductors, Chinese Academy of Sciences, 100083 Beijing, China d College of Materials Science and Opto-Electronics Technology, University of Chinese Academy of Sciences, 100049 Beijing, China e School of Electronic, Electrical, and Communication Engineering, University of Chinese Academy of Sciences, 100049 Beijing, China

# A R T I C L E I N F O

Keywords:   
Range-gated imaging   
3D imaging   
Computational imaging   
Depth estimation   
Depth-prior   
Vision learning

# A B S T R A C T

3D Range-gated Imaging (3DRGI) has great potential for long-range detection in adverse weather and nighttime low-light conditions. However, the performance of traditional optical-based 3DRGI methods is highly dependent on hardware characteristics. Although existing vision-learning-based methods have mitigated these hardware limitations, their accuracy remains constrained. Our recent work has demonstrated that integrating optical depth-prior derived from range-gated imaging into vision-learning-based 3DRGI models can combine the advantages of optical-based and vision-learning-based methods and enhance the accuracy of depth maps. In this paper, we propose three architectural and prior innovations that collectively lead to significant improvements in 3DRGI methods. First, we refine the optical depth-prior to eliminate significant noise caused by multipath effects. Second, we introduce a transformer-based architecture that learns adaptive depth intervals under the supervision of the optical depth-prior to further design an optical-prior guided loss function. Finally, we introduce a depth perception prior and incorporate it into the vision learning model, further enhancing the overall depth estimation accuracy. The proposed method effectively integrates depth cues from multiple depth priors into a transformerbased architecture and significantly enhances the overall accuracy. Comparison experiments and ablation studies were conducted to validate the feasibility of the proposed method, and the results demonstrate that the proposed architecture achieves state-of-the-art performance. The results show that root mean square error achieved $4 5 \%$ to $6 0 \%$ improvement compared with Gated2Depth under harsh weather scenarios.

# 1. Introduction

Long-range three-dimensional (3D) detection in adverse weather or nighttime low-light conditions remains a significant challenge for autonomous drones, robotics and driver assistance systems [1–3]. Most existing fully autonomous vehicles strongly rely on scanning LiDAR for 3D detection. However, current scanning LiDAR is constrained by mechanically limited angular sampling rates, which result in low spatial resolution at extended distances, especially beyond $1 0 0 ~ \mathrm { { m } }$ range. RGB stereo depth estimation methods perform well under favorable weather conditions but often fail in low-light scenarios when no reliable features [4,5]. Moreover, their functionality is severely compromised in harsh weather such as dense fog or heavy rain. Range-gated imagers integrate the transient response from flash-illuminated scenes in broad temporal bins. Its ability to suppress backscatter significantly enhances the detection range and shows robustness to low-light, and adverse weather conditions. In addition, the embedded time-off light information can be decoded as depth.

Traditional 3DRGI methods have been developed from 2004 and are based on the principle of range-gated imaging. They encode depth information into intensity information and depth information is then decoded from the intensity relationships of multiple gating images with specific parameters. So far, traditional optical-based 3DRGI methods can be divided into three categories: time stepping [6–9], gain modulation [10,11], and range-intensity correlation imaging [12–14]. The time slicing method requires processing a large volume of time-sliced gated images to achieve 3D reconstruction, resulting in poor real-time performance. Both gain modulation and range-intensity correlation methods can reconstruct a 3D scene with at least two gate images. Gain modulation methods obtain depth information by modulating the gain of the intensifier in the gated camera and their performance relies on high-performance equipment such as high-stability lasers. Rangeintensity correlation methods assume laser pulses and gate pulses as rectangular and obtain depth from trapezoidal or triangular rangeintensity profiles (RIPs). Traditional optical-based 3DRGI methods have matured and its performance is independent of the scene. However, its performance dependents on hardware characteristics, limiting its flexibility and practicality.

In recent years, with the rise of visual learning technologies, visionlearning-based 3DRGI methods have been developed by some technology companies to approach the 3DRGI problem from a visual perspective. Vision-learning-based 3DRGI is also known as depth estimation from gated images in the field of computer vision. In 2016 Amit Adam et al. [15] in Microsoft Research propose to employ non-parametric regression trees to simultaneously estimate depth map, albedo and ambient light intensity from gated images. In 2019 Tobias Gruber et al. [16] in Daimler AG have proposed firstly to use convolutional neural network to exploit semantic context and regress depth map across gated slices, which firstly realize 3DRGI without any hardware dependence and achieve comparing accuracy with scanning lidar. In 2022 Amanpreet Walia et al. [17] also proposed a self-supervised 3DRGI method with no need for depth ground truth. In 2023 Stefanie Walz et al. [18] in Mercedes-Benz use two gated cameras for joint depth estimation from gated and wide-baseline active stereo cues.

These vision-learning-based 3DRGI methods have brought new directions and perspectives for this area as they overcome the hardware limitations of traditional methods and improve flexibility greatly. However, these methods are totally deduced from visual characteristics of gated images and do not consider optical principle of range-gated imaging, which leads to limited accuracy. Our recent work [19] has proved that integrating depth cues derived from the optical property of range-gated imaging into vision-guided 3DRGI model is helpful to enhance the accuracy of depth map. However, in work [19], the effectiveness of the optical depth prior is constrained by two main factors. First, the depth prior masks contain significant noise caused by multipath effects. Second, in the attempt to integrate depth priors into the vision model, statistical methods were utilized to approximate depth intervals. However, this approach introduced significant estimation errors, ultimately resulting in limited accuracy improvements.

In this paper, we propose three architectural and loss innovations that combined, lead to large improvements in the vision-learning-based

3DRGI methods. First, we introduce a depth perception prior and incorporate it into the vision-learning model, enhancing the overall depth estimation accuracy. Second, we refine the optical depth-prior to eliminate noise caused by multipath effect. Finally, we introduce a transformer-based architecture that learns adaptive depth intervals to further design a optical-prior guided loss function. The proposed method effectively integrates depth cues from multiple depth priors into a transformer-based architecture, thereby significantly enhancing the overall accuracy. The remainder of this paper is organized as follows. In Section 2. Methodology, we describe the proposed method in detail. Comparison experiments and ablation experiments with state-of-the-art 3DRGI methods are discussed in Section 3 and Section 4 gives conclusions.

# 2. Methodology

Fig. 1 illustrates our proposed architecture. Similar to other rangegated 3D imaging methods, our approach takes three gated images as input and produces a refined depth map as output. These three gated images are acquired using the same system parameters, except for the time delay between the laser pulse and the gate pulse. As a result, they capture different but overlapping distance ranges. The backbone is a classification-regression-based depth estimation architecture and adapted from AdaBins [20]. The main contribution of this paper is the introduction of a depth-perception prior and a refined optical depth prior. Additionally, we propose a method for tightly integrating these two priors into a visual learning network, ultimately enhancing overall depth estimation accuracy.

In this section, we first review the key ideas behind depth estimation using adaptive bins and then we focus on our three architectural and loss innovations. We begin by introducing the motivation behind depth perception priors and present how to integrate the depth perception prior into adaptive bins to enhance overall depth estimation accuracy. Subsequently, we introduce how to remove noise caused by multipath effect in the optical depth-prior masks and how to tight couple the refined optical depth-prior into the vision learning model. Lastly, we present the loss functions utilized in the proposed 3DRGI method.

# 2.1. Depth estimation using adaptive bins

AdaBins [20] is a classification-regression-based depth estimation architecture, originally designed for estimating a dense depth map from a single RGB input image. As illustrated in Fig. 1, it consists of an encoder-decoder architecture combined with a transformer-based module mViT, which incorporates global distribution information, thereby enhancing accuracy. This mViT module partitions the depth range into bins, with their center values estimated adaptively for each image. The final depth values are then computed as linear combinations of these bin centers. By transforming the depth regression task into a classification task, AdaBins achieves improved performance. For the detailed network structures of AdaBins and mViT, please refer to Ada-Bins [20].

![](images/b07826fb8f6275f9ac298faa335e0d96075003a326b69dc4578ff41088a7dad1.jpg)  
Fig. 1. Architecture of our proposed method.

During training, three gated images are input into an encoderdecoder architecture to generate decoder features. These features are then fed into the mViT module, which produces adaptive bin widths $^ { b }$ and range attention maps $R$ . We can obtain the depth bin width centers $\boldsymbol { c } ( \boldsymbol { b } ) = \{ c ( b _ { 1 } ) , c ( b _ { 2 } ) , c ( b _ { 3 } ) , . . . . , c ( b _ { N } ) \} _ { }$ , which are derived from the binwidth vector $^ { b }$ as follows:

$$
c ( b _ { i } ) = d _ { m i n } + ( d _ { m a x } - d _ { m i n } ) \bigg ( b _ { i } / 2 + \sum _ { j = 1 } ^ { i - 1 } b _ { j } \bigg )
$$

Range attention maps $R$ are processed through a ${ \bf 1 } \times { \bf 1 }$ convolutional layer to produce $N$ output channels, followed by applying a softmax activation function. Then, we can obtain $U _ { x y } ^ { k }$ where ${ k = 1 , 2 , 3 , . . . . , N }$ which presents the each pixels probabilities in every bin width. Then $c ( b _ { k } )$ and $U _ { x y } ^ { k }$ are used together to obtain the initial depth map $d _ { A d a b i n s }$ :

$$
d _ { A d a B i n s } = \sum _ { k = 1 } ^ { N } c ( b _ { k } ) U _ { x y } ^ { k }
$$

# 2.2. Depth discretization strategy based on depth perception prior

AdaBins divides the depth range into bins, with the center value of each bin estimated adaptively for each image. AdaBins has demonstrated its depth discretization strategy is the best option when compared to the fixed depth discretization strategy. However, the AdaBins strategy does not consider the depth perception prior. Specifically, the imaging system demonstrates higher accuracy in perceiving near-range information, whereas the perception of distant information tends to be less precise and more coarse-grained. Based on this prior knowledge of depth perception, we prefer to allocate more and denser bins in the near range and fewer and sparser bins in the far range. Thus, we propose a depth discretization strategy that integrates the depth perception prior and the depth discretization strategy of AdaBins. Our ablation study validates the effectiveness of the proposed discretization strategy.

Fig. 2illustrates the two choices of bin widths. As shown in Fig. 2, the first choice employs equal-interval discretization in logarithmic space, allocating a higher density of bins in the near range while distributing fewer and sparser bins in the far range. This discretization strategy aligns with the depth perception prior. The second one is from Adabins. Next we will introduce how to integrate the two choices for bin widths.

Assuming that the depth interval $D = ( d _ { m i n } , d _ { m a x } )$ needs to be discretized into $N$ bins, the equal-interval discretization strategy in logarithmic space can be formulated as:

$$
t _ { i } = e ^ { \log ( d _ { m i n } ) } + \frac { \log ( d _ { m a x } / d _ { m i n } ) ^ { * } i } { N }
$$

where $t _ { i } \in \left\{ t _ { 1 } , t _ { 2 } , t _ { 3 } , . . . t _ { N } \right\}$ are discretization thresholds.

AbaBins outputs an N-dimensional vector $^ { b }$ which represents bin widths. We integrate these two strategies by obtaining the proportion $p _ { i }$ of the each interval in the summary first:

$$
p _ { i } = \frac { t _ { i } } { \sum _ { j = 1 } ^ { N } t _ { j } }
$$

Then, we get the $b _ { i } ^ { \prime }$ by computing the dot product of $p _ { i }$ and $b _ { i }$

$$
b _ { i } ^ { \prime } = { b _ { i } } ^ { * } p _ { i }
$$

Finally, we normalize the vector $b _ { i } ^ { \prime }$ so that its elements sum to 1, thereby obtaining the refined bin-width vector $b ^ { \prime \prime }$ as follows:

$$
b _ { i } ^ { \prime \prime } = \frac { b _ { i } ^ { \prime } + \epsilon } { \Sigma _ { j = 1 } ^ { N } \Big ( b _ { j } ^ { \prime } + \epsilon \Big ) } , \in = 1 0 ^ { - 3 }
$$

The small positive $\in$ ensures each bin width is strictly positive. The normalization introduces a competition among the bin-widths and conceptually forces the network to focus on sub-intervals within $D$ by predicting smaller bin-widths at interesting regions of $D$ .

# 2.3. Tight coupling of refined optical depth-prior into vision learning model

Our recent work [19] has demonstrated that integrating depth priors derived from the principle of range-gated imaging into vision-guided 3DRGI models enhances the accuracy of depth maps. However, in existing methods, the effectiveness of the optical depth-prior is constrained by two main factors. First, the depth prior masks contain significant noise caused by multipath effects. Second, in the attempt to integrate depth priors into the vision model, statistical methods were utilized to approximate depth intervals. However, this approach introduced significant estimation errors, ultimately resulting in resulting in limited accuracy improvements. In this section, we begin by introduce how to remove noise caused by multipath effect in the optical depthprior. Then we introduce a transformer-based architecture that learns adaptive depth intervals to integrate the refined optical depth-prior into the vision learning model.

# 2.3.1. Deducing refined optical depth-prior from range-gated imaging

The flow chart for deducing refined depth priors is illustrated in Fig. 3. Next we briefly introduce the principle of the optical depth prior, which is detailed in our recent work [19,21]. According to range-gated imaging, the depth range can be divided into three zones in depth, and they can be described as

![](images/29b692e0e8b3d0950f1e2d13c8400f45d3b39892164029b7fd5736d4fac7c843.jpg)  
Fig. 2. Two choices for bin widths and our method combines the two choices.

![](images/3253b8eea07a59a45b32dc6272dd7cc38dacd81e1f403d04c9903b4524879ecb.jpg)  
Fig. 3. Flow chart for deducing refined optical depth-prior from range-gated imaging. In the optical depth-prior masks, $h ^ { 1 }$ represents the nearest depth zone, $h ^ { 2 }$ represents the intermediate depth zone, and $h ^ { 3 }$ represents the farthest depth zone.

$$
\begin{array} { r l } & { Z _ { x y } ^ { 1 } : = \left\{ ( \mathbf { x } , \mathbf { y } ) | p _ { x y } ^ { 1 } > p _ { x y } ^ { 2 } \land p _ { x y } ^ { 1 } > p _ { x y } ^ { 3 } \right\} } \\ & { Z _ { x y } ^ { 2 } : = \left\{ ( \mathbf { x } , \mathbf { y } ) | p _ { x y } ^ { 2 } > p _ { x y } ^ { 1 } \land p _ { x y } ^ { 2 } > p _ { x y } ^ { 3 } \right\} } \\ & { Z _ { x y } ^ { 3 } : = \left\{ ( \mathbf { x } , \mathbf { y } ) | p _ { x y } ^ { 3 } > p _ { x y } ^ { 1 } \land p _ { x y } ^ { 3 } > p _ { x y } ^ { 2 } \right\} } \end{array}
$$

where $\displaystyle ( \mathbf { x } , \mathbf { y } )$ represents pixel coordinates in the three gated images, $p _ { x y } ^ { i }$ represents intensity value of pixel in $\displaystyle ( \mathbf { x } , \mathbf { y } )$ in the ith gated image where $i \in \{ 1 , 2 , 3 \} . \ Z _ { x y } ^ { 1 }$ represents the nearest zone, $Z _ { x y } ^ { 2 }$ represents the intermediate zone, and $Z _ { x y } ^ { 3 }$ represents the farthest zone.

We also define masks to eliminate saturated pixels and pixels with low variance as follows:

$$
M _ { x y } : = \left\{ ( \boldsymbol { x } , \boldsymbol { y } ) \vert \operatorname* { m a x } _ { i \in \{ 1 , 2 , 3 \} } \Bigl ( p _ { x y } ^ { i } \Bigr ) < \gamma \right\}
$$

$$
B _ { x y } : = \left\{ ( x , y ) | \left( \operatorname* { m a x } _ { i \in \{ 1 , 2 , 3 \} } \left( p _ { x y } ^ { i } \right) - \operatorname* { m i n } _ { i \in \{ 1 , 2 , 3 \} } \left( p _ { x y } ^ { i } \right) \right) > \theta \right\}
$$

However, as illustrated in Fig. 4, the depth prior masks contain significant noise caused by multipath effects. Retro reflective traffic signs reflect the illumination light back towards the camera but spread out the light illuminating other surface causing multipath effects. In automotive scenes, the most severe multipath effects result from reflective road masks surfaces. Multipath effect superposes the low intense ground plane intensity with the high intensity multi-path reflection containing further distant pulse information, leading to a wrong mask for the ground pixels. Building upon Gated2Gated [17], we leverage camera intrinsics to mitigate these noise artifacts. We first estimate a conservative constant ground plane with normal $n$ and height $h$ . Furthermore, we estimate an approximated depth measurement $r$ by comparing the intensity values of the three gated slices. This allows us to filter out pixels $( x , y )$ that get back-projected to 3D coordinates substantially

![](images/a33e27cee04cecdef7c8c8257794053c87b4d74da72cd5a25b466c7970c69aed.jpg)  
Fig. 4. Demonstration of noise caused by multipath effect. The top row displays the three gated images, and the second row shows the optical depth prior masks, where light green represents the nearest depth zone, pink represents the intermediate depth zone, and blue represents the farthest depth zone. It can be observed that in both scenes, the road surface is illuminated by the reflected light from traffic signs, resulting in noise appearing on the road surface in the left image of the second row. These noise points originally belonged to the nearest depth zone but were incorrectly classified as belonging to the farthest depth zone. Our method effectively removed these noise points.

lower than the ground plane:

$$
E _ { x y } : = \{ ( { \pmb x } , { \pmb y } ) | \big ( \hat { r } K ^ { - 1 } x _ { t } \big ) { \pmb n } < h \}
$$

where $\boldsymbol { x } _ { t } = [ x , y , 1 ]$ denote homogeneous pixel coordinates and $K$ denote the camera matrix. Lastly, the refined optical depth-prior masks $h _ { x y }$ are as follows:

$$
\begin{array} { r } { h _ { x y } ^ { i } = \left\{ \begin{array} { l l } { 1 } & { i f \left( x , y \right) \in Z _ { x y } ^ { i } \land M _ { x y } \land B _ { x y } \land E _ { x y } } \\ { 0 } & { o t h e r w i s e } \end{array} \right. , } \\ { i \in \{ 1 , 2 , 3 \} \quad \quad } \end{array}
$$

As shown in the right image of the second row in Fig. 4, our method effectively removed invalid priors caused by multipath effects.

# 2.3.2. Learning adaptive depth intervals from the AdaDepth module

The refined optical depth-prior masks divide the depth range into three parts in depth and can offer depth cues. Integrating depth cues into the vision learning model to enhance accuracy is the next challenge we aim to address. If we can obtain the depth intervals of the three depth zones, we can find out pixels whose depth estimation is inconsistent with the optical depth-prior and compel the model to focus on pixels to improve overall accuracy. In our previous work [19], statistical methods were employed to approximate the depth intervals for the three depth regions. However, this approach introduced significant estimation errors, ultimately resulting in resulting in limited accuracy improvements. In this paper, we introduce a transformer-based architecture that learns adaptive depth intervals under the supervision of the optical depthprior. Next, we will introduce how to learn the adaptive depth intervals from our proposed AdaDepth module under the supervision of the refined optical depth-prior.

The architecture of AdaDepth module is illustrated in Fig. 5. As illustrated in Fig. 5, we use a mViT module to output a 3-dimensional vector $p b$ along with their respective 3-channels range-attention-maps $p R$ . Similarly to AdaBins, $p b$ represent bin-width. The three bin-centers $c ( p b )$ can be expressed as follows:

$$
\begin{array} { c } { c ( p b _ { i } ) = d _ { m i n } + ( d _ { m a x } - d _ { m i n } ) \bigg ( p b _ { i } / 2 + \sum _ { j = 1 } ^ { i - 1 } p b _ { j } \bigg ) , } \\ { i = 1 , 2 , 3 } \end{array}
$$

The depth intervals of the three depth zones can be expressed as:

$$
d _ { 1 } = d _ { m i n } + ( d _ { m a x } - d _ { m i n } ) \left( \frac { p b ^ { 1 } } { p b ^ { 1 } + p b ^ { 2 } + p b ^ { 3 } } \right)
$$

$$
d _ { 2 } = d _ { m i n } + ( d _ { m a x } - d _ { m i n } ) \left( \frac { p b ^ { 1 } + p b ^ { 2 } } { p b ^ { 1 } + p b ^ { 2 } + p b ^ { 3 } } \right)
$$

We expected estimating adaptive depth intervals for each image from the regression. $p U ^ { i }$ and $c ( p b _ { i } )$ are used together to obtain depth map $d _ { A d a D e p t h }$ :

$$
d _ { A d a D e p t h } = \sum _ { k = 1 } ^ { 3 } c ( p b _ { k } ) p U _ { x y } ^ { k }
$$

$d _ { A d a D e p t h }$ are supervised by the ground truth during the training of the model. As shown in Fig. 5, the range-attention-maps $p R$ are passed through a $1 \times 1$ convolutional layer to produce 3-channels, which is followed by a SoftMax activation. The 3 SoftMax score $p U ^ { k }$ , where $k = 1$ , 2, 3 are considered as probabilities over three depth zones. During the training of the model, for each epoch, we can classify all the pixels into three categories by finding the maximum value of the $p U _ { x y } ^ { k }$ vector as follows:

$$
\begin{array} { r } { R _ { x y } ^ { k } = \left\{ \begin{array} { l l } { 1 } & { i f p U _ { x y } ^ { k } = m a x \Big ( p U _ { x y } ^ { 1 } , p U _ { x y } ^ { 2 } , p U _ { x y } ^ { 3 } \Big ) } \\ { 0 } & { o t h e r w i s e } \end{array} \right. , } \\ { k \in \{ 1 , 2 , 3 \} } \end{array}
$$

Equation (16) means the model predicts the corresponding pixel belongs to the $k t h$ depth zone. If the prediction is inconsistent with the optical-prior, we will penalize the confidence of the corresponding pixel:

$$
\begin{array} { r } { G _ { x y } = \left\{ \begin{array} { c c } { C _ { x y } , } & { i f \ h _ { x y } ^ { i } \wedge R _ { x y } ^ { i } = 1 \ , } \\ { C _ { x y } - \mu ^ { \ast } m a x \Big ( p U _ { x y } ^ { k } \Big ) , } & { o t h e r w i s e } \end{array} \right. } \\ { i \in \{ 1 , 2 , 3 \} , k \in \{ 1 , 2 , 3 \} } \end{array}
$$

where $\mu$ denotes the hyperparameters, and $C _ { x y }$ represents the confidence map, which indicates the model’s confidence in both in $p b$ and $p R ,$ as well as the output from the mViT module. The confidence map is utilized to adaptively fuse $d _ { A d a B i n s }$ and $d _ { A d a D e p t h }$ to generate the final depth prediction:

![](images/de52bdccd82111fd1417029e8a2d0cd46fa612a4a0b702ccfefd86ad97a72cca.jpg)  
Fig. 5. Architecture of the AdaDepth module.

$$
d _ { x y } = \left( 1 - C _ { x y } \right) \odot d _ { A d a b i n s } + C _ { x y } \odot d _ { A d a D e p t h }
$$

where $d _ { x y }$ is the final depth map and supervised by the ground truth during the training of the model. Equation (17) penalizes predictions that is inconsistent with the optical depth prior. In this way, we learn adaptive depth intervals under the supervision of the optical depth prior.

# 2.3.3. Designing optical prior-guided weight

In addition to utilizing the optical depth prior to supervise the learning of adaptive depth intervals for the three depth regions, we also developed an optical depth prior-guided loss function, which leverages optical depth-prior and the learned depth intervals to further integrate the optical depth prior into the vision learning model. Here, we introduce how to calculate the optical depth prior-guided weight map based on optical depth-prior and the learned depth intervals, and the optical depth prior-guided loss function will be given in section 2.4 Loss function.

By comparing the pixel depth values with the depth boundary values of their respective regions, pixels that exceed these boundaries can be identified, revealing inconsistencies between the depth estimations and the priors. For such pixels, we assign larger weights to compel the model to focus on them and improve overall accuracy. Firstly, we mark $W _ { x y }$ as the weight map. $W _ { x y }$ has the same size with the input gated images and all pixels in $W _ { x y }$ are initially set to 1. For Zone1, pixels whose depth estimates are bigger than $d _ { 1 }$ are inconsistent with the optical depth-prior masks and are assigned additional weights. The corresponding weight map can be expressed as:

$$
\widetilde { W } _ { 1 } = \left\{ \begin{array} { c c } { { d ( x , y ) - d _ { 1 } , \ h _ { x y } ^ { 1 } \wedge d ( x , y ) > d _ { 1 } } } \\ { { 0 , \qquad o t h e r w i s e } } \end{array} \right.
$$

Here, $d ( x , y )$ represents the predicted depth value at the pixel coordinate $( x , y )$ within the depth map estimated at each epoch. $d _ { 1 }$ and $d _ { 2 }$ are predicted from AdaDepth model, and $h _ { x y } ^ { i }$ where $i = { 1 , 2 , 3 }$ , represents three depth prior masks. For Zone2, pixels whose depth estimates are smaller than $d _ { 1 }$ or bigger than $d _ { 2 }$ are inconsistent with the depth prior. The additional weight map is:

$$
\widetilde { W } _ { 2 } = \left\{ \begin{array} { c c } { \displaystyle { d _ { 1 } - d ( x , y ) , ~ h _ { x y } ^ { 2 } \wedge d ( x , y ) < d _ { 1 } } } \\ { \displaystyle { d ( x , y ) - d _ { 2 } , ~ h _ { x y } ^ { 2 } \wedge d ( x , y ) > d _ { 2 } } } \\ { 0 , ~ o t h e r w i s e } \end{array} \right.
$$

For Zone3, pixels whose depth estimates are smaller than $d _ { 2 }$ are inconsistent with the depth prior. The additional weight map is:

$$
\widetilde { W } _ { 3 } = \left\{ \begin{array} { c c } { { d _ { 2 } - d ( x , y ) , \ h _ { x y } ^ { 3 } \wedge d ( x , y ) < d _ { 2 } } } \\ { { 0 , \qquad o t h e r w i s e } } \end{array} \right.
$$

The final refined weight map can be described as:

$$
W _ { x y } ' = W _ { x y } + \sum _ { i = 1 } ^ { 3 } \widetilde { W } _ { i }
$$

# 2.4. Loss function

The optical prior-guided loss can be described as:

$$
\mathcal { L } _ { o p t i c a l - d e p t h - p r i o r } = \sqrt { \frac { 1 } { T } \sum _ { i } g _ { \it x y } ^ { 2 } - \frac { 1 } { T ^ { 2 } } \bigg ( \sum _ { i } g _ { \it x y } \bigg ) ^ { 2 } }
$$

where $g _ { x y } = \left( l o g \widetilde { d } _ { x y } - l o g d _ { x y } \right) \odot W _ { x y }$ ʹ, $\widetilde { d } _ { x y }$ represents the ground truth.

We also impose a supervision on $d _ { A d a D e p t h }$ in our model by optimizing the following loss:

$$
\mathcal { L } _ { A d a D e p t h } = \sqrt { \frac { 1 } { T } { \sum _ { i } } { { m ^ { 2 } } _ { x y } } - \frac { 1 } { T ^ { 2 } } { \left( { \sum _ { i } } { { m _ { x y } } } \right) ^ { 2 } } }
$$

where $m _ { x y } = \bigg ( l o g \widetilde { d } _ { x y } - l o g d _ { A d a D e p t h ( x y ) } \bigg )$ , this approach guides the model to focus on estimating the depth intervals.

We also incorporated the $\mathcal { L } _ { c h a m f e r }$ loss [22] as follows:

$$
\mathcal { L } _ { c h a m f e r } = c h a m f e r ( X , c ( b ) ) + c h a m f e r ( c ( b ) , X )
$$

where the $c ( b )$ represents the set of bin centers and the $X$ represents the set of all depth values in the ground truth image. This chamfer loss term promotes the alignment of the bin center distribution with the distribution of depth values in the ground truth. Lastly, the total loss is as follows

$$
\mathcal { L } = \lambda \bullet \mathcal { L } _ { o p t i c a l - d e p t h - p r i o r } + \alpha \bullet \mathcal { L } _ { c h a m f e r } + \gamma \mathcal { L } _ { A d a D e p t h }
$$

where the $\lambda , \alpha$ and $\gamma$ are hyperparameters.

# 3. Experiments

In this section, we evaluate our approach using two open-source datasets: the Gated2depth [16] dataset and Seeing through Fog [23] dataset. In the following, we first briefly describe the datasets and the evaluation metrics. For a thorough assessment of our proposed method, we benchmarked the performance against recent studies using the Gated2Depth dataset. Next, an evaluation on adverse weather conditions is conducted using the Seeing through Fog dataset. We also conducted two ablation studies to evaluate the individual effects of two priors and to assess two methods of obtaining optical depth-prior and conduct experiments to evaluate the computing resource consumption and computing time.

# 3.1. Dataset

# 3.1.1. Gated2depth dataset

Gated2depth dataset is the first long-range gated dataset, covering snow, rain, urban and suburban driving during $4 0 0 0 ~ \mathrm { k m }$ in-the-wild acquisition. The Gated2depth datasets were acquired using test vehicles outfitted with high-resolution RGB stereo cameras, scanning LiDAR, and a gated camera with an $8 0 8 ~ \mathrm { n m }$ laser flood-light source integrated into the front bumper. The gated camera operates at a framerate of 120 Hz, which was split up in three slices plus an additional ambient capture without active illumination. The datasets span multiple cities across Germany, Denmark, and Sweden, totaling 17,686 image frames. The Gated2depth dataset also includes a synthetic dataset, which was adapted from the GTA5 simulator, and is structured to mirror real-world conditions, facilitating the production of authentic gated measurements. It includes 8,157 samples designated for training and 1,647 samples allocated for testing.

# 3.1.2. Seeing through fog dataset

The Seeing Through Fog dataset is to address the challenges posed by adverse weather conditions. It includes 12,000 samples from real-world driving environments and 1,500 samples gathered in a controlled fog chamber. This dataset encompasses different weather scenarios, including fog, snow, and rain, and was collected over more than 10,000 km of driving in northern Europe. Due to the infrequency of extreme weather conditions, this dataset is not well-suited for training [23], so the method needs to be trained on clean data for evaluation on this dataset.

# 3.2. Evaluation metrics

We use the standard metrics proposed in prior work [24] to evaluate our method, namely RMSE, MAE, ARD and $\delta _ { i } < 1 . 2 5 ^ { i } f o r i \in ( 1 , 2 , 3 )$ . The metrics are evaluated for distances between $3 \textrm { m }$ and $8 0 ~ \mathrm { { m } }$ , with the range limited by the maximum capabilities of the scanning LiDAR.

# 3.3. Experiments detail

In proposed architectures, taking three gated images as input and produces a depth map as output. It is worth noting that for some architectures used for monocular depth estimation, we conduct experiments with two modalities for comparison. The optical depth-prior mask is generated from three gated images and used to compute the proposed loss function. The training process is supervised by the depth ground truth generated by scanning LiDAR.

Our networks were implemented using Pytorch1.13. We trained the model with an NVIDIA 4090 24 GB GPU. We use ADAM optimizer with the learning rate set to 0.0001.

# 3.4. Evaluation

# 3.4.1. Evaluation on Gated2depth dataset

In the comparative study, to thoroughly assess the performance of our method, we extended our comparison to various state-of-the-art monocular depth estimation methods, such as the transformer-based DPT [25], with AdaBins serving as our baseline, a classificationregression-based depth estimation approach. These methods, initially developed for predicting on RGB datasets, were separately evaluated on RGB and gated data provided by the Gated2Depth dataset. We also conducted evaluations with 3DRGI models tailored for range-gated imaging scenarios, including Gated2Depth [16], the pioneering visionguided 3DRGI model, the self-supervised learning framework Gated2- Gated [17], and our previous work[19], which is the first 3DRGI model incorporating optical property of range-gated imaging into vision learning model.

We first assessed the performance of all methods on the Gated2Depth test dataset, utilizing sparse LiDAR points as validation data. The results are detailed in Table 1 and Fig. 6. It is important to note that in Table 1 the top-performing results for each category are emphasized in bold, and the second-best results are underlined. This highlighting convention is consistently used in Table 2, Table 3 and Table 4 and Table 5 as well.

From Table 1 and Fig. 1, it is evident that our method has achieved comprehensive performance improvements compared to existing methods, which demonstrate the effectiveness of our method. As shown in Table 1, both during the day and at night, we outperform our baseline model, AdaBins, in all metrics, indicating the effectiveness of the two priors we proposed. It is noteworthy that both AdaBins and DPT are trained and evaluated on RGB and gated images, with their performance metrics being superior when trained and evaluated on gated images rather than RGB. Particularly at night, the RMSE improvement ranges from $2 4 \%$ to $3 5 \%$ , highlighting the superiority of range-gated imaging in adverse weather and nighttime low-light conditions over RGB.

Besides, compared with direct regression-based depth models, such as Gated2depth and Gao’s work, classification-regression-based architectures, such as AdaBins, perform excellently in terms of RMSE and MAE, but do not surpass on the ARD and $\delta _ { i }$ metrics. The results demonstrate that classification-regression-based architectures achieve superior performance in terms of overall accuracy and long-range depth estimation; however, their performance diminishes in regions with smaller depths, as also can be observed in Fig. 6. Our proposed method is also a classification-regression-based architecture. We address this issue by incorporating our depth perception prior, which allocates more bins at closer distances, thereby increasing accuracy at short ranges. Experimental results in Table 1 show that our method not only performs excellently in terms of RMSE and MAE, but also achieves comparable results on the ARD and $\delta _ { i }$ metrics. We will provide additional details in the ablation study of different priors.

# 3.4.2. Evaluation on seeing through fog dataset

To evaluate our model performance under adverse weather conditions, we test our method on the Seeing Through Fog dataset. Our utilizes the DROR [26] filtering algorithm, which removes $8 . 2 ~ \%$ of the LiDAR points to minimize clutter in the LiDAR ground truth.

Table 1 Evaluation results of the proposed and existing method on gated2depth dataset.   

<table><tr><td>Method</td><td>Modality</td><td>RMSE↓</td><td>ARD↓</td><td>MAE↓</td><td>81</td><td>82↑</td><td>83</td></tr><tr><td colspan="8">Day</td></tr><tr><td>Dpt [25]</td><td>gated</td><td>6.88</td><td>0.17</td><td></td><td>83.08</td><td>92.10</td><td>95.31</td></tr><tr><td>Dpt [25]</td><td>rgb</td><td>7.65</td><td>0.16</td><td></td><td>79.61</td><td>92.04</td><td>96.61</td></tr><tr><td>AdaBins [20]</td><td>gated</td><td>6.15</td><td>0.20</td><td>3.47</td><td>75.05</td><td>91.94</td><td>95.94</td></tr><tr><td>AdaBins [20]</td><td>rgb</td><td>10.90</td><td>0.22</td><td>4.41</td><td>75.59</td><td>89.16</td><td>94.58</td></tr><tr><td>Gated2gated [17]</td><td>gated</td><td>8.46</td><td>0.17</td><td>4.37</td><td>83.56</td><td>93.12</td><td>96.09</td></tr><tr><td>Gated2depth [16]</td><td>gated</td><td>7.61</td><td>0.12</td><td>3.53</td><td>88.07</td><td>94.32</td><td>96.60</td></tr><tr><td>Gao [19]</td><td>gated</td><td>7.19</td><td>0.12</td><td>3.28</td><td>89.11</td><td>94.98</td><td>96.90</td></tr><tr><td>Ours</td><td>gated</td><td>5.87</td><td>0.15</td><td>3.26</td><td>84.93</td><td>93.37</td><td>96.36</td></tr><tr><td colspan="8">Night</td></tr><tr><td>Dpt [25]</td><td>gated</td><td>7.00</td><td>0.18</td><td></td><td>81.90</td><td>91.84</td><td>95.22</td></tr><tr><td>Dpt [25]</td><td>rgb</td><td>9.19</td><td>0.20</td><td></td><td>76.90</td><td>90.07</td><td>94.59</td></tr><tr><td>AdaBins [20]</td><td>gated</td><td>6.35</td><td>0.17</td><td>3.72</td><td>74.68</td><td>91.93</td><td>95.87</td></tr><tr><td>AdaBins [20]</td><td>rgb</td><td>9.73</td><td>0.20</td><td>4.67</td><td>77.40</td><td>91.82</td><td>96.36</td></tr><tr><td>Gated2gated [17]</td><td>gated</td><td>9.43</td><td>0.21</td><td>4.86</td><td>82.17</td><td>91.54</td><td>94.48</td></tr><tr><td>Gated2depth [16]</td><td>gated</td><td>8.39</td><td>0.15</td><td>3.79</td><td>87.52</td><td>93.00</td><td>95.21</td></tr><tr><td>Gao [19]</td><td>gated</td><td>8.15</td><td>0.15</td><td>3.71</td><td>87.89</td><td>93.39</td><td>95.36</td></tr><tr><td>Ours</td><td>gated</td><td>6.07</td><td>0.15</td><td>3.59</td><td>84.86</td><td>94.31</td><td>95.94</td></tr></table>

As shown in Table 2, our method outperforms other approaches under adverse weather conditions. Specifically, under harsh weather scenarios, our method achieves RMSE improvements of $4 5 ~ \%$ to $6 0 ~ \%$ compared to Gated2Depth. Additionally, compared to our previous methods, we have achieved enhancements of approximately $2 0 \%$ to 40 $\%$ , demonstrates our method robustness under adverse conditions.

Under extreme weather conditions (e.g., dense fog with visibility below $5 ~ \mathrm { m } )$ ), the absence of reliable ground-truth depth weakens the supervision signal, thereby limiting the performance of our network. We acknowledge this as a limitation and will explore potential improvements in future work.

# 3.4.3. Experiment of resource consumption and time consumption

To assess the resource consumption and runtime performance of our method, we conduct a comparative analysis on the Gated2depth dataset. We conduct a comparative analysis of resource and time consumption on the Gated2depth dataset and the results are shown in Table 3. The results presented in Table 3 are averaged across 1786 gated frames from the Gated2Depth test dataset. In Table 3, the “Processing Speed” item indicates how many frames the model can process per second during the inference stage; the “GPU Memory” item refers to the maximum GPU memory consumption during inference; and the “GPU” item specifies the GPU model used during testing. In Table 3, results for all models except Gated2Depth were obtained using an RTX 4090 GPU, while the results for the Gated2Depth model were directly cited from [16] due to its official implementation being based on TensorFlow 1.0, which is incompatible with execution on the RTX 4090. As shown in Table 3, the Gated2Depth model benefits from its simple network architecture to achieve the highest processing speed, reaching 25 FPS on a single Titan V GPU, even though the Titan V is considerably less capable than the RTX 4090. Our method reaches about $1 8 ~ \mathrm { F P S }$ , which is slightly lower than our baseline AdaBins. This is primarily due to the additional network components introduced for learning depth prior information, which increase resource usage and time consumption. Gated2gated shows the worst performance in resource consumption, primarily because its architecture employs three separate convolutional networks, resulting in exceptionally high computational demands.

![](images/583328752468b2fed030dc4137b73e3b85783678ddd5a95b92e5df0e175f4716.jpg)  
Fig. 6. Each example is organized as follows: the top row displays the gated image, consisting of three gated slices captured for each instance. The second ro presents the corresponding RGB image. The third row shows the LiDAR point cloud in the gated view, while the final row features the depth maps produced b different models.

Table 2 Evaluation results of the proposed and existing methods on seeing through fog dataset.   

<table><tr><td rowspan="2"></td><td rowspan="2">Methode</td><td colspan="4">Clear</td><td colspan="4">Light fog</td></tr><tr><td>RMSE↓</td><td>MAE↓</td><td>δ2↑</td><td>83↑</td><td>RMSE↓</td><td>MAE↓</td><td>82↑</td><td>83↑</td></tr><tr><td rowspan="5">Night</td><td>Gao [19]</td><td>7.05</td><td>3.23</td><td>94.33</td><td>96.43</td><td>7.36</td><td>3.80</td><td>90.82</td><td>93.99</td></tr><tr><td>Gated2gated [17]</td><td>11.69</td><td>6.74</td><td>89.58</td><td>92.83</td><td>11.29</td><td>6.46</td><td>89.31</td><td>93.17</td></tr><tr><td>Gated2depth [16]</td><td>10.06</td><td>5.17</td><td>90.59</td><td>93.39</td><td>9.94</td><td>5.37</td><td>89.80</td><td>93.63</td></tr><tr><td>AdaBins [20]</td><td>6.24</td><td>3.12</td><td>95.50</td><td>98.00</td><td>6.94</td><td>3.45</td><td>93.60</td><td>97.30</td></tr><tr><td>Ours</td><td>5.63</td><td>3.07</td><td>94.20</td><td>96.80</td><td>6.17</td><td>3.23</td><td>91.60</td><td>95.50</td></tr><tr><td rowspan="6">Day</td><td>Gao [19]</td><td>6.90</td><td>3.28</td><td>94.70</td><td>96.93</td><td>6.23</td><td>2.93</td><td>94.62</td><td>96.95</td></tr><tr><td>Gated2gated [17]</td><td>11.15</td><td>6.31</td><td>90.48</td><td>93.97</td><td>10.70</td><td>6.01</td><td>91.52</td><td>91.52</td></tr><tr><td>Gated2depth [16]</td><td>11.48</td><td>6.60</td><td>87.38</td><td>91.58</td><td>11.28</td><td>6.63</td><td>88.66</td><td>92.56</td></tr><tr><td>AdaBins [20]</td><td>6.02</td><td>3.23</td><td>95.50</td><td>98.00</td><td>4.90</td><td>2.84</td><td>96.00</td><td>96.00</td></tr><tr><td>Ours</td><td>5.40 dense fog</td><td>3.10</td><td>95.10</td><td>98.50</td><td>4.68</td><td>2.78</td><td>93.60</td><td>97.90</td></tr><tr><td colspan="3"></td><td></td><td colspan="4"></td><td></td></tr><tr><td rowspan="5">Night</td><td>Gao [19]</td><td>7.21</td><td>3.60</td><td>91.82</td><td>93.81</td><td>snow 8.78</td><td>4.49</td><td>90.82</td><td>93.99</td></tr><tr><td>Gated2gated [17]</td><td>13.52</td><td>8.69</td><td>86.70</td><td>90.61</td><td>11.91</td><td>6.80</td><td>90.09</td><td>93.31</td></tr><tr><td>Gated2depth [16]</td><td>12.51</td><td>7.72</td><td>86.59</td><td>90.81</td><td>10.70</td><td>5.81</td><td>89.45</td><td>93.02</td></tr><tr><td>AdaBins [20]</td><td>5.45</td><td>3.32</td><td>96.80</td><td>98.10</td><td>7.14</td><td>4.15</td><td>91.60</td><td>95.50</td></tr><tr><td>Ours</td><td>4.40</td><td>3.13</td><td>96.00</td><td>97.30</td><td>5.49</td><td>3.90</td><td>94.00</td><td>96.40</td></tr><tr><td rowspan="5">Day</td><td>Gao [19]</td><td>7.34</td><td>4.08</td><td>91.59</td><td>93.79</td><td>10.15</td><td>4.86</td><td>91.19</td><td>94.34</td></tr><tr><td>Gated2gated [17]</td><td>11.09</td><td>6.86</td><td>91.43</td><td>94.47</td><td>10.97</td><td>6.28</td><td>91.12</td><td>94.63</td></tr><tr><td>Gated2depth [16]</td><td>11.86</td><td>7.85</td><td>87.10</td><td>91.70</td><td>11.28</td><td>6.61</td><td>87.93</td><td>92.50</td></tr><tr><td>AdaBins [20]</td><td>4.97</td><td>3.59</td><td>93.60</td><td>97.00</td><td>6.39</td><td>4.11</td><td>91.60</td><td>95.80</td></tr><tr><td>Ours</td><td>4.39</td><td>3.32</td><td>92.90</td><td>96.70</td><td>6.03</td><td>3.48</td><td>91.70</td><td>96.50</td></tr></table>

Table 3 Resource consumption and time consumption of different methods are evaluated on the gated2dpeth datasets.   

<table><tr><td>Model</td><td>Processing speed (FPS)↑</td><td>GPU memory (MB)↓</td><td>GPU</td></tr><tr><td>Gated2depth [16]</td><td>25</td><td>1</td><td>Titan V</td></tr><tr><td>Gated2gated [17]</td><td>2.70</td><td>17,914</td><td>RTX4090</td></tr><tr><td>AdaBins[20]</td><td>19.29</td><td>565</td><td>RTX4090</td></tr><tr><td>Ours</td><td>18.22</td><td>759</td><td>RTX4090</td></tr></table>

Table 4 Ablation results of the proposed priors on gated2depth dataset.   

<table><tr><td></td><td>Prior</td><td>RMSE↓</td><td>81↑</td><td>82↑</td><td>83↑</td><td>ARD↓</td></tr><tr><td>DAY</td><td>None</td><td>6.15</td><td>75.05</td><td>91.94</td><td>95.54</td><td>0.20</td></tr><tr><td></td><td>Optcial depth prior</td><td>6.03</td><td>76.63</td><td>92.76</td><td>96.39</td><td>0.19</td></tr><tr><td></td><td>Depth perception prior</td><td>6.07</td><td>83.40</td><td>92.50</td><td>96.04</td><td>0.15</td></tr><tr><td></td><td>Both priors included</td><td>5.89</td><td>84.93</td><td>93.37</td><td>96.36</td><td>0.15</td></tr><tr><td>NIGHT</td><td>None</td><td>6.35</td><td>74.68</td><td>91.93</td><td>95.94</td><td>0.17</td></tr><tr><td></td><td>Optcial depth prior</td><td>6.13</td><td>80.64</td><td>91.34</td><td>95.21</td><td>0.17</td></tr><tr><td></td><td>Depth perception prior</td><td>6.16</td><td>81.26</td><td>92.71</td><td>95.98</td><td>0.15</td></tr><tr><td></td><td>Both priors included</td><td>6.07</td><td>83.86</td><td>92.41</td><td>95.87</td><td>0.15</td></tr></table>

Table 5 Ablation results of methods obtaining the optical prior- depth intervals on gated2depth dataset.   

<table><tr><td></td><td>Method</td><td>RMSE ←</td><td>81↑</td><td>8↑</td><td>8↑</td><td>ARD ←</td></tr><tr><td>DAY</td><td>Statistical method</td><td>5.97</td><td>80.36</td><td>93.27</td><td>96.32</td><td>0.15</td></tr><tr><td></td><td>Vision learning method</td><td>5.89</td><td>84.93</td><td>93.37</td><td>96.36</td><td>0.15</td></tr><tr><td>NIGHT</td><td>Statistical method</td><td>6.22</td><td>76.73</td><td>92.81</td><td>96.11</td><td>0.16</td></tr><tr><td></td><td>Vision learning method</td><td>6.07</td><td>83.86</td><td>92.41</td><td>95.87</td><td>0.15</td></tr></table>

It should be noted that, beyond the complexity of the model itself, computational resource consumption and computation time are also dependent on the performance of the GPU used. Although the computation time of our proposed method is higher than that of the Gated2Depth model, it has the potential to achieve real-time depth estimation performance when deployed on more powerful GPUs.

# 3.4.4. Ablation experiment – different priors

To assess the independent contributions of each prior, we conducted an ablation study, with the results presented in Table 4. For the baseline method, we observed the poorest performance, whereas incorporating either the optical depth prior or depth perception prior significantly improved the RMSE scores. Training with these two priors synergistically enhanced the overall performance metrics.

It can be observed in Table 4 that the incorporation of depth perception prior alone leads to a substantial improvement in the ARD metric. This is because our depth perception prior allocates more bins at closer distances, thereby increasing accuracy at short ranges, thus improving the ARD.

# 3.4.5. Ablation experiment – different methods for deducing depth intervals of the three depth zones

Accurate depth intervals of the three depth zones are crucial for the optical depth prior-guided loss function. In our previous work [19], we utilized statistical method to derive depth intervals. In the paper, we introduce a transformer-based architecture to learn adaptively depth intervals under the supervision of the refined optical depth-prior. To assess the performance of both methods, we conducted an ablation study, and the results are summarized in Table 5. As observed from Table 5, the vision learning method generally outperforms the static method, demonstrating the effectiveness of our proposed approach.

# 4. Conclusion

In this work, we presented an approach to improving the accuracy of 3DRGI by integrating refined optical depth-prior and a depth discretization strategy based on depth-perception prior within a transformer-based architecture. Our method effectively addresses key challenges such as multipath effect and weak depth prior integration, leading to a significant enhancement in depth estimation accuracy. Through extensive comparison experiments and ablation studies, we demonstrated that our proposed framework achieves state-of-the-art performance, significantly reducing RMSE under both adverse and clear conditions.

The experimental results confirm the robustness of our method, proving its effectiveness in maintaining accurate depth estimation even in challenging environments. Looking forward, future work will focus on further refining the integration of optical properties and depth priors, as well as exploring more advanced vision learning models to enhance accuracy in complex real-world scenarios.

# CRediT authorship contribution statement

Xiaoquan Liu: Writing – review & editing, Validation, Supervision. Di Zhang: Writing – review & editing, Writing – original draft, Methodology, Investigation. Jinming Gao: Validation, Software, Investigation. Xinwei Wang: Supervision, Project administration, Investigation.

# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Acknowledgements

The authors acknowledge the financial funding of this work and thank the creators of the Gated2Depth and Seeing through Fog datasets for their valuable contributions. This work was supported by the National Natural Science Foundation of China (Grant No.62305256), the Natural Science Foundation of Hubei Province (GrantNo.2022CFC038) and the Science Foundation Research Project of Wuhan Institute of Technology (Grant No. K2021054),

# Data availability

The data is publicly available online

# References

[1] Q.A. Al-Haija, M. Gharaibeh, A. Odeh, Detection in adverse weather conditions for autonomous vehicles via deep learning, AI 3(2) (2022) 303-317 [Online]. Available: https://www.mdpi.com/2673-2688/3/2/19.   
[2] D. Kumar, N. Muhammad, object detection in adverse weather for autonomous driving through data merging and YOLOv8, Sensors, vol. 23, no. 20, p. 8471, 2023. [Online]. Available: https://www.mdpi.com/1424-8220/23/20/8471.   
[3] W. Liu, G. Ren, R. Yu, S. Guo, J. Zhu, L. Zhang, Image-adaptive YOLO for object detection in adverse weather conditions, Proceedings of the AAAI Conference on Artificial Intelligence 36 (2) (2022) 1792–1800.   
[4] H. Laga, L.V. Jospin, F. Boussaid, M. Bennamoun, A survey on deep learning techniques for stereo-based depth estimation, IEEE Trans. Pattern Anal. Mach. Intell. 44 (4) (2020) 1738–1764.   
[5] A. Badki, A. Troccoli, K. Kim, J. Kautz, P. Sen, O. Gallo, Bi3d: Stereo depth estimation via binary classifications, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 1597–1605.   
[6] J. Busck, H. Heiselberg, Gated viewing and high-accuracy three-dimensional laser radar, Appl. Opt. 43 (24) (2004) 4705–4710.   
[7] P. Andersson, Long-range three-dimensional imaging using range-gated laser radar images, Opt. Eng. 45 (2006) 4301, doi: 10.1117/1.2183668.   
[8] J. Busck, Underwater 3-D optical imaging with a gated viewing laser radar, Optical Engineering 44 (2005) 116001, doi: 10.1117/1.2127895.   
[9] J. Andersen, J. Busck, H. Heiselberg, Long distance high accuracy 3-D laser radar and person identification, Proc SPIE 5791 (2005) 05/19, https://doi.org/10.1117/ 12.604345.   
[10] Z. Xiuda, Y. Huimin, J. Yanbing, Pulse-shape-free method for long-range threedimensional active imaging with high linear accuracy, Opt. Lett. 33 (11) (2008) 1219–1221.   
[11] C. Jin, X. Sun, Y. Zhao, Y. Zhang, L. Liu, Gain-modulated three-dimensional active imaging with depth-independent depth accuracy, Opt. Lett. 34 (22) (2009) 3550–3552, https://doi.org/10.1364/OL.34.003550.   
[12] M. Laurenzis, F. Christnacher, D. Monnin, Long-range three-dimensional active imaging with superresolution depth mapping, Opt. Lett, 32 (21) (2007) 3146–3148.   
[13] W. Xinwei, L. Youfu, Z. Yan, Triangular-range-intensity profile spatial-correlation method for 3D super-resolution range-gated imaging, Appl. Opt. 52 (30) (2013) 7399–7406.   
[14] P. Wang, H. Liu, S. Qiu, Y. Liu, F. Huang, Three-dimensional super-resolution range-gated imaging based on Gaussian-range-intensity model, Appl. Opt. 62 (29) (2023) 7633–7642.   
[15] A. Adam, C. Dann, O. Yair, S. Mazor, S. Nowozin, Bayesian time-of-flight for realtime shape, illumination and albedo, IEEE Trans. Pattern Anal. Mach. Intell. 39 (5) (2016) 851–864.   
[16] T. Gruber, F. Julca-Aguilar, M. Bijelic, F. Heide, Gated2depth: Real-time dense lidar from gated images, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, Seoul, Korea (South), 2019, pp. 1506-1516.   
[17] A. Walia et al., Gated2gated: Self-supervised depth estimation from gated images, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, LA, USA, 2022, pp. 2801-2811.   
[18] S. Walz, M. Bijelic, A. Ramazzina, A. Walia, F. Mannan, F. Heide, Gated stereo: Joint depth estimation from gated and wide-baseline active stereo cues, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Vancouver, BC, Canada, 2023, pp. 13252-13262.   
[19] J. Gao, X. Liu, Z. Yang, Depth-prior-based range-gated 3D imaging method: integration of optical property into visual learning, Opt. Express 32 (22) (2024) 39355–39368, https://doi.org/10.1364/OE.531362.   
[20] S.F. Bhat, I. Alhashim, P. Wonka, Adabins: Depth estimation using adaptive bins, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, Nashville, TN, USA, 2021, pp. 4008-4017.   
[21] Z. Yang, X. Liu, Depth-prior-based LiDAR point cloud de-noising method leveraging range-gated imaging, Opt. Lett. 49 (18) (2024) 5212–5215, https://doi. org/10.1364/OL.530278.   
[22] H. Fan, H. Su, L. Guibas, A point set generation network for 3D object reconstruction from a single image, in: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017, pp. 2463–2471.   
[23] M. Bijelic, et al., Seeing through fog without seeing fog: Deep multimodal sensor fusion in unseen adverse weather, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11679–11689.   
[24] D. Eigen, C. Puhrsch, R. Fergus, Depth map prediction from a single image using a multi-scale deep network, in: presented at the Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2, Montreal, Canada, 2014.   
[25] R. Ranftl, A. Bochkovskiy, V. Koltun, Vision transformers for dense prediction, in: Proceedings of the IEEE/CVF international conference on computer vision, Montreal, QC, Canada, 2021, pp. 12159–12168.   
[26] N. Charron, S. Phillips, S.L. Waslander, De-noising of lidar point clouds corrupted by snowfall, in: 2018 15th Conference on Computer and Robot Vision (CRV), Toronto, ON, Canada, 2018, IEEE, pp. 254–261.