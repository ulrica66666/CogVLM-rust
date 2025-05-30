// 实现一个基本的 Transformer Block（LayerNorm + Attention + FFN）。
use ndarray::{Array2};
use crate::rope::apply_rope;

pub struct TransformerBlock {
    pub embed_dim: usize,
    // 模拟权重
    pub attn_weight: Array2<f32>,
    pub mlp_weight: Array2<f32>,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize) -> Self {
        let attn_weight = Array2::<f32>::eye(embed_dim);
        let mlp_weight = Array2::<f32>::eye(embed_dim);
        TransformerBlock { embed_dim, attn_weight, mlp_weight }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut x_rope = x.clone();
        apply_rope(&mut x_rope, self.embed_dim);
        let attn_out = x_rope.dot(&self.attn_weight);
        let mlp_out = attn_out.dot(&self.mlp_weight);
        mlp_out
    }
}
