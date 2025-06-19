use ndarray::{Array2, Axis, Zip, s};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

pub struct LayerNorm {
    pub epsilon: f32,
    pub gamma: Array2<f32>, // (1, dim)
    pub beta: Array2<f32>,  // (1, dim)
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        LayerNorm {
            epsilon: 1e-5,
            gamma: Array2::ones((1, dim)),
            beta: Array2::zeros((1, dim)),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), 0.0);
        let mut normalized = x.clone();

        Zip::from(normalized.rows_mut())
            .and(&mean)
            .and(&var)
            .for_each(|mut row, &m, &v| {
                row -= m;
                row /= (v + self.epsilon).sqrt();
            });

        // 逐元素乘gamma+beta
        normalized * &self.gamma + &self.beta
    }
}

// 激活函数gelu
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
}

fn gelu_array(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(gelu)
}

fn scaled_dot_product_attention(
    q: &Array2<f32>, 
    k: &Array2<f32>, 
    v: &Array2<f32>, 
) -> Array2<f32> {
    let dk = q.shape()[1] as f32;
    let scores = q.dot(&k.t()) / dk.sqrt();
    
    let mut exp_scores = scores.mapv(f32::exp);
    for mut row in exp_scores.outer_iter_mut() {
        let sum = row.sum();
        if sum > 0.0 {
            row /= sum;
        }
    }

    exp_scores.dot(v)
}

// 多头自注意力
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub wq: Array2<f32>, // (embed_dim, num_heads * head_dim)
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        let dist = Uniform::new(-0.1, 0.1);

        MultiHeadAttention {
            num_heads,
            head_dim,
            wq: Array2::random((embed_dim, embed_dim), dist),
            wk: Array2::random((embed_dim, embed_dim), dist),
            wv: Array2::random((embed_dim, embed_dim), dist),
            wo: Array2::random((embed_dim, embed_dim), dist),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // x (seq_len, embed_dim)
        let seq_len = x.shape()[0];
        let embed_dim = x.shape()[1];
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        // QKV shape(seq_len, embed_dim)
        let q = x.dot(&self.wq);
        let k = x.dot(&self.wk);
        let v = x.dot(&self.wv);

        // 按head分割 重新reshape (num_heads, seq_len, head_dim)
        let q = q.into_shape((seq_len, num_heads, head_dim)).unwrap();
        let k = k.into_shape((seq_len, num_heads, head_dim)).unwrap();
        let v = v.into_shape((seq_len, num_heads, head_dim)).unwrap();

        // 每个头计算attention
        let mut heads_out = Vec::with_capacity(num_heads);
        for head_idx in 0..num_heads {
            let q_head = q.slice(s![.., head_idx, ..]).to_owned();
            let k_head = k.slice(s![.., head_idx, ..]).to_owned();
            let v_head = v.slice(s![.., head_idx, ..]).to_owned();

            let attn_out = scaled_dot_product_attention(&q_head, &k_head, &v_head);
            heads_out.push(attn_out);
        }

        // 拼接head输出(seq_len, embed_dim)
        let mut concat = Array2::<f32>::zeros((seq_len, embed_dim));
        for (head_idx, head_out) in heads_out.into_iter().enumerate() {
            let start = head_idx * head_dim;
            let end = start + head_dim;
            concat.slice_mut(s![.., start..end]).assign(&head_out);
        }

        // 输出线性层
        concat.dot(&self.wo)
    }
}

// 前馈网络
pub struct FeedForward {
    pub w1: Array2<f32>, // (embed_dim, ff_dim)
    pub w2: Array2<f32>, // (ff_dim, embed_dim)
    pub b1: Array2<f32>, // (1, ff_dim)
    pub b2: Array2<f32>, // (1, embed_dim)
}

impl FeedForward {
    pub fn new(embed_dim: usize, ff_dim: usize) -> Self {
        let dist = Uniform::new(-0.1, 0.1);
        FeedForward {
            w1: Array2::random((embed_dim, ff_dim), dist),
            w2: Array2::random((ff_dim, embed_dim), dist),
            b1: Array2::zeros((1, ff_dim)),
            b2: Array2::zeros((1, embed_dim)),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let hidden = gelu_array(&(x.dot(&self.w1) + &self.b1));
        hidden.dot(&self.w2) + &self.b2
    }
}

// Transformer 层
pub struct TransformerLayer {
    pub embed_dim: usize,
    pub ff_dim: usize,
    pub num_heads: usize,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub mha: MultiHeadAttention,
    pub ffn: FeedForward,
}

impl TransformerLayer {
    pub fn new(embed_dim: usize, ff_dim: usize, num_heads: usize) -> Self {
        TransformerLayer {
            embed_dim,
            ff_dim,
            num_heads,
            ln1: LayerNorm::new(embed_dim),
            ln2: LayerNorm::new(embed_dim),
            mha: MultiHeadAttention::new(embed_dim, num_heads),
            ffn: FeedForward::new(embed_dim, ff_dim),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let x_norm = self.ln1.forward(x);
        let attn_out = self.mha.forward(&x_norm);
        let x = x + attn_out; 

        let x_norm = self.ln2.forward(&x);
        let ffn_out = self.ffn.forward(&x_norm);
        x + ffn_out 
    }
}
