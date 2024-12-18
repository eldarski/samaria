#include "attention.h"
#include "error_handling.h"

namespace jtext
{
    namespace models
    {
        torch::Tensor MultiHeadAttention::forward(const torch::Tensor &x)
        {
            try
            {
                auto batch_size = x.size(0);
                auto seq_length = x.size(1);

                // Linear projections and reshape for multi-head
                auto q = reshape_for_attention(query_->forward(x));
                auto k = reshape_for_attention(key_->forward(x));
                auto v = reshape_for_attention(value_->forward(x));

                // Scaled dot-product attention
                auto scores = torch::matmul(q, k.transpose(-2, -1));
                scores = scores / std::sqrt(head_dim_);

                // Apply attention mask if sequence length > 1
                if (seq_length > 1)
                {
                    auto mask = create_attention_mask(seq_length, x.device());
                    scores = scores.masked_fill(mask == 0, -1e9);
                }

                auto attn = torch::softmax(scores, -1);
                auto context = torch::matmul(attn, v);

                // Reshape and project back
                context = context.transpose(1, 2)
                              .contiguous()
                              .view({batch_size, -1, num_heads_ * head_dim_});

                return output_->forward(context);
            }
            catch (const std::exception &e)
            {
                throw ModelError("MultiHeadAttention forward pass failed: " + std::string(e.what()));
            }
        }

        torch::Tensor MultiHeadAttention::create_attention_mask(int seq_length, torch::Device device)
        {
            auto mask = torch::ones({seq_length, seq_length}, device);
            mask = torch::triu(mask, /*diagonal=*/1);
            return mask.unsqueeze(0).unsqueeze(0);
        }
    }
} // namespace jtext