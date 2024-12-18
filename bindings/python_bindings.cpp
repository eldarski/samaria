#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include "contrastive_model.h"
#include "image_processor.h"
#include "japanese_text_processor.h"

namespace py = pybind11;

std::vector<float> tensor_to_vector(const torch::Tensor& tensor) {
    auto t = tensor.contiguous().cpu();
    std::vector<float> result;
    auto flat = t.view({-1});
    auto accessor = flat.accessor<float, 1>();
    result.reserve(flat.numel());
    for (int64_t i = 0; i < flat.numel(); i++) {
        result.push_back(accessor[i]);
    }
    return result;
}

PYBIND11_MODULE(_samaria_bindings, m) {
    m.doc() = "Japanese text and image contrastive learning library";

    py::class_<samaria::JapaneseTextProcessor>(m, "JapaneseTextProcessor")
        .def(py::init<>())
        .def("tokenize", &samaria::JapaneseTextProcessor::tokenize)
        .def("tokenize_batch", &samaria::JapaneseTextProcessor::tokenize_batch)
        .def("generate_embeddings",
             [](samaria::JapaneseTextProcessor& self, const std::string& text) {
                 return tensor_to_vector(self.generate_embeddings(text));
             })
        .def("generate_embeddings_batch",
             [](samaria::JapaneseTextProcessor& self, const std::vector<std::string>& texts,
                int batch_size) {
                 return tensor_to_vector(self.generate_embeddings_batch(texts, batch_size));
             });

    py::class_<samaria::ImageProcessor>(m, "ImageProcessor")
        .def(py::init<>())
        .def("extract_features",
             [](samaria::ImageProcessor& self, const std::string& path) {
                 return tensor_to_vector(self.extract_features(path));
             })
        .def("extract_features_batch",
             [](samaria::ImageProcessor& self, const std::vector<std::string>& paths) {
                 return tensor_to_vector(self.extract_features_batch(paths));
             })
        .def("get_description", &samaria::ImageProcessor::get_description);

    py::class_<samaria::ContrastiveModel>(m, "ContrastiveModel")
        .def(py::init<int>(), py::arg("embedding_dim") = 256)
        .def("compute_similarity",
             [](samaria::ContrastiveModel& self, const py::array_t<float>& text_emb,
                const py::array_t<float>& img_emb) {
                 // Convert numpy arrays to torch tensors
                 auto text_tensor = torch::from_blob((void*)text_emb.data(), {1, text_emb.size()},
                                                     torch::TensorOptions().dtype(torch::kFloat32))
                                        .clone();

                 auto img_tensor = torch::from_blob((void*)img_emb.data(), {1, img_emb.size()},
                                                    torch::TensorOptions().dtype(torch::kFloat32))
                                       .clone();

                 return self.compute_similarity(text_tensor, img_tensor);
             })
        .def("train", &samaria::ContrastiveModel::train)
        .def("train_batch", &samaria::ContrastiveModel::train_batch);
}