/*
 * Author : Lintang E
 *          Fahri Wahyu P
 *          Robby A
 * Created 2 May 2019 08:30 AM
 * Updated 23 May 2019 03:29
 */
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

class FuzzyCMeans{
private:
    float **membership;
    float **last_membership;
    vector<pair<int, int> > cluster;
    vector<vector<Point> > cluster_points;
    Mat *center_vectors;
    float m_const;
    Mat data;
    int num_steps;
    int data_dimension;
    int num_of_clusters;
    int num_of_data;
    float error_threshold;
    Vec3b* color;
public:
    FuzzyCMeans(const Mat &data, int num_of_clusters, int num_steps, float m_const, float error_threshold){
        this->data = data.clone();
        this->num_of_data = this->data.rows * this->data.cols;
        this->data_dimension = this->data.channels();
        this->num_of_clusters = num_of_clusters;
        this->num_steps = num_steps;
        this->m_const = m_const;
        this->error_threshold = error_threshold;

        membership = new float*[num_of_clusters];
        last_membership = new float*[num_of_clusters];
//        float uniform_membership = 1.0f/num_of_clusters;
        float init_membership[num_of_clusters][num_of_data];
//        float total_membership=0.0;
        for(int i=0;i<num_of_clusters;i++){
            for(int j=0;j<num_of_data;j++){
                init_membership[i][j] = (1.0f + (float)(rand() % 99))/100.0f;
//                total_membership += init_membership[i][j];
            }
        }

        for(int i=0;i < num_of_clusters; i++){
            membership[i] = new float[num_of_data];
            last_membership[i] = new float[num_of_data];
            for(int j=0;j < num_of_data;j++){
//                init_membership[i][j] /= total_membership;
                membership[i][j] = init_membership[i][j];
//                std::cout << membership[i][j] << std::endl;
                last_membership[i][j] = init_membership[i][j];
            }
        }

        center_vectors = new Mat[num_of_clusters];        
        color = new Vec3b[num_of_clusters];

        cluster.resize(num_of_clusters);
        cluster_points.resize(num_of_clusters);
        for(int i=0;i<num_of_clusters;i++){
            center_vectors[i] = (Mat_<float>(3,1) << rand()%255, rand()%255, rand()%255);//Mat::zeros(data_dimension,1,CV_32FC1);
            cluster[i].first = 0;
            cluster[i].second = 0;
            color[i] = {(uchar)(rand()%255), (uchar)(rand()%255), (uchar)(rand()%255)};
        }

    }

    ~FuzzyCMeans(){

        //TODO : delete the allocate memory
    }

    inline Mat pixToMat(const Vec3b &_pix){
        return (Mat_<float>(3,1) << _pix[0],_pix[1],_pix[2]);
    }

    inline float calcNorm(const Mat &_data){
        float a = _data.at<float>(0);
        float b = _data.at<float>(1);
        float c = _data.at<float>(2);
        return sqrt(a*a + b*b + c*c);
    }

//    inline float calcCost(int x, int y){

//    }

    Mat process(){
        float exp_arg = 2.0f/(m_const - 1.0f);
        float last_error = 0.0;
        int step = 0;
        for(step=0;step < num_steps; step++){
            float err = 0.0;
            cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
            for(int cluster=0;cluster < num_of_clusters;cluster++){
                Mat num = Mat::zeros(3,1,CV_32FC1);
                float den = .0f;
                for(int idx=0;idx < num_of_data;idx++){
                    float temp = std::pow(membership[cluster][idx], m_const);
                    num += temp * pixToMat(data.at<Vec3b>(idx));
                    den += temp;
                }
                center_vectors[cluster] = (1.0f/den) * num;
                //DEBUGGING ONLY
                cout << "-----------------------------------------------------" << endl;
                cout << "Class : " << (cluster+1) << endl;
                cout << "Den : " << den << " ; Num : " << num << endl;
                cout << "Center Vectors : " << center_vectors[cluster] << endl;
            }

            float term1 = .0f;
            float term2 = .0f;
            for(int cluster=0;cluster < num_of_clusters;cluster++){
                for(int idx = 0; idx < num_of_data; idx++){
                    float per_idx = .0f;
                    Mat temp1 = pixToMat(data.at<Vec3b>(idx));
                    Mat temp2 = temp1 - center_vectors[cluster];
                    term1 = calcNorm(temp2);
                    for(int cluster2 = 0; cluster2 < num_of_clusters; cluster2++){
//                        if(cluster2==cluster)continue;
                        Mat temp3 = temp1 - center_vectors[cluster2];
                        term2 = calcNorm(temp3);
//                        cout << "Temp 1 : " << temp1 << endl;
//                        cout << "Temp 2 : " << temp2 << endl;
//                        cout << "Temp 3 : " << temp3 << endl;
//                        cout << "Term 1 : " << term1 << endl;
//                        cout << "Term 2 : " << term2 << endl;
//                        cout << endl;
                        per_idx += std::pow(term1/(term2+1e-6), exp_arg);
                    }
                    per_idx = 1.0f / (per_idx+1e-6);
//                    std::cout << per_idx << " ; " << last_membership[cluster][idx] << std::endl;
                    membership[cluster][idx] = per_idx;
                    err += std::pow(per_idx - last_membership[cluster][idx], 2.0f);
                    last_membership[cluster][idx] = per_idx;
                }
            }

            err = sqrt(err);
            //DEBUGGING ONLY
            cout << "Error after " << (step+1) << " steps : " << err << endl;
            cout << endl;
            last_error = err;
            if(err < error_threshold)
                break;

        }
//        cout << "Error after " << step << " steps : " << last_error << endl;
        Mat result = Mat(data.size(), CV_8UC3);
        for(int idx = 0;idx < num_of_data; idx++){
            float best_cost = .0f;
            int idx_cluster = 0;
            Mat pix_mat = pixToMat(data.at<Vec3b>(idx));
            for(int cluster=0;cluster<num_of_clusters;cluster++){
                Mat temp = pix_mat  - center_vectors[cluster];
                float temp2 = calcNorm(temp);
                float cost = membership[cluster][idx]*pow(temp2,2);
                if(cost > best_cost){
                    best_cost = cost;
                    idx_cluster = cluster;
                }
            }
            cluster[idx_cluster].first += calcNorm(pix_mat);
            cluster[idx_cluster].second++;
            cluster_points[idx_cluster].emplace_back(Point(idx%data.cols,idx/data.cols));
            result.at<Vec3b>(idx) = color[idx_cluster];
        }
        return result;
    }

    Mat getHighestAvg(){
        float best_avg = 0;
        int best_cluster_idx=0;
        for(int i=0;i<num_of_clusters;i++){
            float avg = cluster[i].first/cluster[i].second;
            if(avg > best_avg){
                best_avg = avg;
                best_cluster_idx = i;
            }
        }
        Mat draw = Mat::zeros(data.size(), CV_8UC1);
        for(vector<Point>::iterator it = cluster_points[best_cluster_idx].begin();
            it!= cluster_points[best_cluster_idx].end(); it++){
            draw.at<uchar>(it->y, it->x) = 255;
        }
//        drawContours(draw,cluster_points,best_cluster_idx,Scalar(255),CV_FILLED);
        return draw;
    }
};

enum ChannelCode{
    RED = 0b00000001,
    GREEN = 0b00000010,
    BLUE = 0b00000100,
};

Mat extractOneChannel(Mat &_input, uchar mask){
    Mat output = Mat::zeros(_input.size(), CV_8UC3);
    uchar red_mask = 0b00000001 & mask;
    uchar green_mask = 0b00000010 & mask;
    uchar blue_mask = 0b00000100 & mask;
    for(int i=0;i<_input.rows;i++){
        Vec3b* in_ptr = _input.ptr<Vec3b>(i);
        Vec3b* out_ptr = output.ptr<Vec3b>(i);
        for(int j=0;j<_input.cols;j++){
            out_ptr[j][0] = in_ptr[j][0] * blue_mask;
            out_ptr[j][1] = in_ptr[j][1] * green_mask;
            out_ptr[j][2] = in_ptr[j][2] * red_mask;
//            out_ptr[j][0] = max(in_ptr[j][0],max(in_ptr[j][1],in_ptr[j][2]));
//            out_ptr[j][1] = 0;
//            out_ptr[j][2] = 0;
        }
    }
    return output;
}

int main(){
    srand(time(0));
    Mat input_img = imread("/media/koseng/4A7AE1C07AE1A941/lit/tpc/240px-Fundus_photograph_of_normal_right_eye.jpg");

    FuzzyCMeans clustering(input_img, 5, 100, 1.001f, .001f);

    Mat result = clustering.process();
    Mat result2 = clustering.getHighestAvg();

    imshow("INPUT", input_img);
    imshow("RESULT1", result);
    imshow("RESULT2", result2);

    waitKey(0);

//    for(int i=0;i <30;i++){
//        stringstream ss;
//        ss << "/home/koseng/Downloads/dataset/im" << setfill('0') << setw(4) << i+1 << ".ppm";
////        ss << "/home/koseng/Downloads/Data/image" << i+1 << "prime" << ".tif";
//        cout << "Input Image : " << i+1 << endl;
//        Mat input_img = imread(ss.str());
//        resize(input_img,input_img,Size(200,200));
//        FuzzyCMeans clustering(input_img, 5, 100, 1.001f, .001f);
//        Mat result = clustering.process();
//        Mat result2 = clustering.getHighestAvg();
//        stringstream file1; file1 << "/home/koseng/Downloads/dataset/result/input_resized" << i+1 << ".jpg";
//        stringstream file2; file2 << "/home/koseng/Downloads/dataset/result/segmented" << i+1 << ".jpg";
//        stringstream file3; file3 << "/home/koseng/Downloads/dataset/result/contour" << i+1 << ".jpg";
//        imwrite(file1.str(),input_img);
//        imwrite(file2.str(),result);
//        imwrite(file3.str(),result2);

////        imshow("INPUT", input_img);
////        imshow("RESULT1", result);
////        imshow("RESULT2", result2);
//    }

//    waitKey(0);
    return 0;
}
