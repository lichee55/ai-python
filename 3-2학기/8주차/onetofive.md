### 20171181 최영빈

-   1부터 5까지의 합 tf version1 형식으로 작성

```python
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

with tf.compat.v1.Session() as sess:
    total = tf.Variable(0)
    sess.run(total.initializer)

    for i in range(1, 6):
        total = total.assign_add(i)

    print('tf1으로 실행한 1부터 5까지의 합 : ', total.eval())

```

    tf1으로 실행한 1부터 5까지의 합 :  15


    2021-10-10 18:46:44.365573: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2021-10-10 18:46:44.365598: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
