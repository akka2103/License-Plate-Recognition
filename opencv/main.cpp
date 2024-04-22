#include "opencv2/opencv.hpp"
#include <iostream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <fstream>
#include "main.h"

#define MIN_AR 1        // Minimum aspect ratio
#define MAX_AR 6        // Maximum aspect ratio
#define KEEP 5          // Limit the number of license plates
#define RECT_DIFF 2000  // Set the difference between contour and rectangle

// Random generator for cv::Scalar
cv::RNG rng(12345);

bool compareContourAreas (std::vector<cv::Point>& contour1, std::vector<cv::Point>& contour2) 
{
    const double i = fabs(contourArea(cv::Mat(contour1)));
    const double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}

void LicensePlate::grayscale(cv::Mat& frame) 
{
  cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
}

void LicensePlate::drawLicensePlate(cv::Mat& frame, std::vector<std::vector<cv::Point>>& candidates) 
{
  const int width = frame.cols;
  const int height = frame.rows;
  const float ratio_width = width / (float) 512;    // WARNING! Aspect ratio may affect the performance (TO DO LIST)
  const float ratio_height = height / (float) 512;  // WARNING! Aspect ratio may affect the performance

  // Convert to rectangle and also filter out the non-rectangle-shape.
  std::vector<cv::Rect> rectangles;
  for (std::vector<cv::Point> currentCandidate : candidates) 
  {
    cv::Rect temp = cv::boundingRect(currentCandidate);
    float difference = temp.area() - cv::contourArea(currentCandidate);
    if (difference < RECT_DIFF) {
      rectangles.push_back(temp);
    }
  }

  // Remove rectangle with bad shape.
  rectangles.erase(std::remove_if(rectangles.begin(), rectangles.end(), [](cv::Rect temp) 
  {
    const float aspect_ratio = temp.width / (float) temp.height;
    return aspect_ratio < MIN_AR || aspect_ratio > MAX_AR;
  }), rectangles.end());

  for (cv::Rect rectangle : rectangles) 
  {
    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    cv::rectangle(frame, cv::Point(rectangle.x * ratio_width, rectangle.y * ratio_height), cv::Point((rectangle.x + rectangle.width) * ratio_width, (rectangle.y + rectangle.height) * ratio_height), color, 3, cv::LINE_8, 0);

    // Extract the license plate region
    cv::Mat licensePlateRegion = frame(cv::Rect(rectangle.x * ratio_width, rectangle.y * ratio_height, rectangle.width * ratio_width, rectangle.height * ratio_height));

    // Preprocess the license plate region
    cv::Mat preprocessedRegion;
    cv::resize(licensePlateRegion, preprocessedRegion, cv::Size(200, 50));
    cv::cvtColor(preprocessedRegion, preprocessedRegion, cv::COLOR_BGR2GRAY);
    cv::threshold(preprocessedRegion, preprocessedRegion, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::medianBlur(preprocessedRegion, preprocessedRegion, 3);

    // Perform OCR using Tesseract
    tesseract::TessBaseAPI ocr;
    ocr.Init(nullptr, "eng", tesseract::OEM_LSTM_ONLY);
    ocr.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    ocr.SetImage(preprocessedRegion.data, preprocessedRegion.cols, preprocessedRegion.rows, 1, preprocessedRegion.step);
    std::string licensePlateNumber = std::string(ocr.GetUTF8Text());

    // Postprocess the OCR result
    licensePlateNumber.erase(std::remove_if(licensePlateNumber.begin(), licensePlateNumber.end(), [](unsigned char c) { return !std::isalnum(c); }), licensePlateNumber.end());

    // Append the license plate number to a file
    std::ofstream file("license_plates.txt", std::ios::app);
    if (file.is_open()) {
      file << licensePlateNumber << std::endl;
      file.close();
    }
  }
}

std::vector<std::vector<cv::Point>> LicensePlate::locateCandidates(cv::Mat& frame) 
{
  // Reduce the image dimension to process
  cv::Mat processedFrame = frame;
  cv::resize(frame, processedFrame, cv::Size(512, 512));

  // Must be converted to grayscale
  if (frame.channels() == 3) 
  {
    LicensePlate::grayscale(processedFrame);
  }

  // Perform blackhat morphological operation, reveal dark regions on light backgrounds
  cv::Mat blackhatFrame;
  cv::Mat rectangleKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5)); // Shapes are set 13 pixels wide by 5 pixels tall
  cv::morphologyEx(processedFrame, blackhatFrame, cv::MORPH_BLACKHAT, rectangleKernel);

  // Find license plate based on whiteness property
  cv::Mat lightFrame;
  cv::Mat squareKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(processedFrame, lightFrame, cv::MORPH_CLOSE, squareKernel);
  cv::threshold(lightFrame, lightFrame, 0, 255, cv::THRESH_OTSU);

  // Compute Sobel gradient representation from blackhat using 32 float,
  // and then convert it back to normal [0, 255]
  cv::Mat gradX;
  double minVal, maxVal;
  int dx = 1, dy = 0, ddepth = CV_32F, ksize = -1;
  cv::Sobel(blackhatFrame, gradX, ddepth, dx, dy, ksize); // Looks coarse if imshow, because the range is high?
  gradX = cv::abs(gradX);
  cv::minMaxLoc(gradX, &minVal, &maxVal);
  gradX = 255 * ((gradX - minVal) / (maxVal - minVal));
  gradX.convertTo(gradX, CV_8U);

  // Blur the gradient result, and apply closing operation
  cv::GaussianBlur(gradX, gradX, cv::Size(5,5), 0);
  cv::morphologyEx(gradX, gradX, cv::MORPH_CLOSE, rectangleKernel);
  cv::threshold(gradX, gradX, 0, 255, cv::THRESH_OTSU);

  // Erode and dilate
  cv::erode(gradX, gradX, 2);
  cv::dilate(gradX, gradX, 2);

  // Bitwise AND between threshold result and light regions
  cv::bitwise_and(gradX, gradX, lightFrame);
  cv::dilate(gradX, gradX, 2);
  cv::erode(gradX, gradX, 1);

  // Find contours in the thresholded image and sort by size
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(gradX, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::sort(contours.begin(), contours.end(), compareContourAreas);
  std::vector<std::vector<cv::Point>> top_contours;
  top_contours.assign(contours.end() - KEEP, contours.end()); // Descending order

  return top_contours;
}

void LicensePlate::viewer(const cv::Mat& frame, std::string title) 
{
  cv::imshow(title, frame);
}

int main( int argc, char** argv ) 
{
  // Instantiate LicensePlate object
  LicensePlate lp;

  std::string filename = "002.jpg";
  cv::Mat image;
  image = cv::imread(filename, cv::IMREAD_COLOR);
  if(!image.data ) {
    std::cout <<  "Image not found or unable to open" << std::endl ;
    return -1;
  }
  std::vector<std::vector<cv::Point>> candidates = lp.locateCandidates(image);
  lp.drawLicensePlate(image, candidates);
  lp.viewer(image, "Frame");
  cv::waitKey(0);

  return 0;
}
