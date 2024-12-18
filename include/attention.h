#pragma once
#include <torch/torch.h>

namespace jtext
{
    namespace models
    {
        class MultiHeadAttention : public torch::nn::Module
        {
        public:
            MultiHeadAttention(int hidden_dim, int num_heads)
                : query_(torch::nn::Linear(hidden_dim, hidden_dim)),
                  key_(torch::nn::Linear(hidden_dim, hidden_dim)),
                  value_(torch::nn::Linear(hidden_dim, hidden_dim)),
                  output_(torch::nn::Linear(hidden_dim, hidden_dim)),
                  num_heads_(num_heads),
                  head_dim_(hidden_dim / num_heads)
            {
                register_module("query", query_);
                register_module("key", key_);
                register_module("value", value_);
                register_module("output", output_);
            }

            torch::Tensor forward(const torch::Tensor &x)
            {
                auto batch_size = x.size(0);

                // Linear projections and reshape for multi-head
                auto q = reshape_for_attention(query_->forward(x));
                auto k = reshape_for_attention(key_->forward(x));
                auto v = reshape_for_attention(value_->forward(x));

                // Scaled dot-product attention
                auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(head_dim_);
                auto attn = torch::softmax(scores, -1);
                auto context = torch::matmul(attn, v);

                // Reshape and project back
                context = context.transpose(1, 2)
                              .contiguous()
                              .view({batch_size, -1, num_heads_ * head_dim_});

                return output_->forward(context);
            }

        private:
            torch::nn::Linear query_{nullptr}, key_{nullptr}, value_{nullptr}, output_{nullptr};
            int num_heads_, head_dim_;

            torch::Tensor reshape_for_attention(const torch::Tensor &x)
            {
                auto batch_size = x.size(0);
                return x.view({batch_size, -1, num_heads_, head_dim_})
                    .transpose(1, 2);
            }
        };
    }
}