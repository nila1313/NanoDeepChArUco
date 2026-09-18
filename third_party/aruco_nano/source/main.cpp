
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include "aruco_nano.h"
#include "json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::runtime_error(
                "Usage: detect_batch IMAGE_DIR OUTPUT_JSON DICTIONARY_ID");

        fs::path directory(argv[1]), output(argv[2]);
        std::string dictionary_arg(argv[3]);
        cv::aruco::PredefinedDictionaryType type;
        if (dictionary_arg == "0")
            type = cv::aruco::DICT_4X4_50;
        else if (dictionary_arg == "10")
            type = cv::aruco::DICT_6X6_250;
        else
            throw std::runtime_error("Supported dictionary IDs: 0, 10");

        if (!fs::is_directory(directory))
            throw std::runtime_error("Image directory does not exist");
        if (fs::exists(output))
            throw std::runtime_error("Refusing to overwrite output");

        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(directory))
            if (entry.is_regular_file() && entry.path().extension() == ".jpg")
                files.push_back(entry.path());
        std::sort(files.begin(), files.end());
        if (files.empty())
            throw std::runtime_error("No JPEG images found");

        cv::setNumThreads(1);
        const aruco_nano::ArucoDetector detector(
            cv::aruco::getPredefinedDictionary(type));
        json result = json::object();

        for (const auto& path : files) {
            auto gray = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
            if (gray.empty())
                throw std::runtime_error("Cannot decode " + path.string());

            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            detector.detectMarkers(gray, corners, ids);
            if (ids.size() != corners.size())
                throw std::runtime_error("ID/corner count mismatch");

            json markers = json::object();
            for (std::size_t i = 0; i < ids.size(); ++i) {
                auto key = std::to_string(ids[i]);
                if (markers.contains(key))
                    throw std::runtime_error("Duplicate marker ID");
                if (corners[i].size() != 4)
                    throw std::runtime_error("Invalid marker corner count");
                json points = json::array();
                for (const auto& p : corners[i])
                    points.push_back({p.x, p.y});
                markers[key] = points;
            }
            result[path.filename().string()] = markers;
        }

        std::ofstream stream(output);
        if (!stream)
            throw std::runtime_error("Cannot create output");
        stream << result.dump(2) << '\n';
        stream.close();
        if (!stream)
            throw std::runtime_error("Failed writing output");
        std::cout << "Processed " << files.size()
                  << " JPEG images; OpenCV " << CV_VERSION << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
