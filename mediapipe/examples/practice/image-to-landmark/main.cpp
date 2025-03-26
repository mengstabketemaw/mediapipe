#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "absl/strings/str_cat.h"

// Change these names if your graph uses different stream names.
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "multi_face_landmarks";

// JNI wrapper function: accepts an encoded image as a byte array and returns a JSON string with face mesh data.
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_FaceMesh_detectFaceMesh(JNIEnv *env, jobject /* this */, jbyteArray imageBytes) {
  // Convert the jbyteArray to a std::vector<uchar>.
  jsize len = env->GetArrayLength(imageBytes);
  std::vector<uchar> data(len);
  env->GetByteArrayRegion(imageBytes, 0, len, reinterpret_cast<jbyte*>(data.data()));

  // Decode the image using OpenCV (assumes the image is encoded, e.g. JPEG).
  cv::Mat inputImage = cv::imdecode(data, cv::IMREAD_COLOR);
  if (inputImage.empty()) {
    return env->NewStringUTF("Error: could not decode image");
  }
  
  // Convert the image from BGR (OpenCV default) to RGB (MediaPipe default).
  cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);

  // Wrap the cv::Mat into a MediaPipe ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, inputImage.cols, inputImage.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  inputImage.copyTo(input_frame_mat);

  // Load the MediaPipe graph configuration.
  // (Make sure the config file exists at this location, or adjust the path as needed.)
  std::string graph_config_contents;
  std::string config_file = "mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt";
  auto status = mediapipe::file::GetContents(config_file, &graph_config_contents);
  if (!status.ok()) {
    return env->NewStringUTF("Error: could not load graph config");
  }
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_config_contents);

  // Initialize the MediaPipe CalculatorGraph.
  mediapipe::CalculatorGraph graph;
  status = graph.Initialize(config);
  if (!status.ok()) {
    return env->NewStringUTF("Error: failed to initialize graph");
  }

  // Add an output stream poller for the face mesh landmarks.
  mediapipe::OutputStreamPoller poller;
  status = graph.AddOutputStreamPoller(kOutputStream, &poller);
  if (!status.ok()) {
    return env->NewStringUTF("Error: failed to add output stream poller");
  }

  // Start running the graph.
  status = graph.StartRun({});
  if (!status.ok()) {
    return env->NewStringUTF("Error: failed to start graph");
  }

  // Create a timestamp for the frame.
  size_t frame_timestamp_us =
      static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);

  // Send the image packet into the graph.
  status = graph.AddPacketToInputStream(
      kInputStream,
      mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))
  );
  if (!status.ok()) {
    return env->NewStringUTF("Error: failed to add packet to input stream");
  }

  // Retrieve the output packet from the poller.
  mediapipe::Packet packet;
  if (!poller.Next(&packet)) {
    return env->NewStringUTF("Error: no output packet received");
  }

  // Extract the face mesh landmarks.
  // (Assumes the graph outputs a vector of NormalizedLandmarkList on kOutputStream.)
  const auto& output_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
  for (size_t i = 0; i < output_landmarks.size(); ++i) {
        std::cout << "Landmark List " << i << ":\n";
        const auto& landmark_list = output_landmarks[i];

        for (int j = 0; j < landmark_list.landmark_size(); ++j) {
            const auto& landmark = landmark_list.landmark(j);
            std::cout << "  Landmark " << j << ": ("
                        << landmark.x() << ", " 
                        << landmark.y() << ", " 
                        << landmark.z() << ")\n";
        }
    }

  // Format the landmarks into a JSON string.
  std::string result = "[";
  for (int i = 0; i < output_landmarks.size(); ++i) {
    const auto& landmark_list = output_landmarks[i];
    result += "{\"face_index\": " + std::to_string(i) + ", \"landmarks\": [";
    for (int j = 0; j < landmark_list.landmark_size(); ++j) {
      const auto& lm = landmark_list.landmark(j);
      result += "{\"x\": " + std::to_string(lm.x()) +
                ", \"y\": " + std::to_string(lm.y()) +
                ", \"z\": " + std::to_string(lm.z()) + "}";
      if (j < landmark_list.landmark_size() - 1) {
        result += ", ";
      }
    }
    result += "]}";
    if (i < output_landmarks.size() - 1) {
      result += ", ";
    }
  }
  result += "]";

  // Cleanly shut down the graph.
  graph.CloseInputStream(kInputStream);
  graph.WaitUntilDone();

  // Return the JSON string to Java.
  return env->NewStringUTF(result.c_str());
}
