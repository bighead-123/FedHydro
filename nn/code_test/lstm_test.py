

import tensorflow as tf
import nn

if __name__ == '__main__':
    def gradientTape_test():
        print("一元梯度")
        x = tf.constant(value=3.0)
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            tape.watch(x)
            y1 = 2 * x
            y2 = x * x + 2
            y3 = x * x + 2 * x
        # 一阶导数
        dy1_dx = tape.gradient(target=y1, sources=x)
        dy2_dx = tape.gradient(target=y2, sources=x)
        dy3_dx = tape.gradient(target=y3, sources=x)
        print("dy1_dx:", dy1_dx)
        print("dy2_dx:", dy2_dx)
        print("dy3_dx:", dy3_dx)
    gradientTape_test()

