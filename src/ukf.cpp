#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  n_radar_ = 3;
  n_lidar_ = 2;
  lambda_ = 3 - n_x_;
  time_us_ = 0;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // initialize weights
  weights_ = VectorXd::Zero(2 * n_aug_ + 1);

  weights_[0] = lambda_ / (lambda_ + n_aug_);
  for (size_t i = 1; i < weights_.size(); ++i)
    weights_[i] = 1 / (2 * (lambda_ + n_aug_));

  R_radar_ = MatrixXd::Zero(n_radar_, n_radar_);
  R_radar_(0, 0) = std_radr_ * std_radr_;
  R_radar_(1, 1) = std_radphi_ * std_radphi_;
  R_radar_(2, 2) = std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd::Zero(n_lidar_, n_lidar_);
  R_lidar_(0, 0) = std_laspx_ * std_laspx_;
  R_lidar_(1, 1) = std_laspy_ * std_laspy_;

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
}

UKF::~UKF() = default;

void UKF::ProcessMeasurement(const MeasurementPackage &meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_) {
    P_ = MatrixXd::Identity(n_x_, n_x_);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // set the state with the initial location and zero velocity
      x_ << meas_package.raw_measurements_[0],
          meas_package.raw_measurements_[1],
          0,
          0,
          0;
    } else {
      double rho = meas_package.raw_measurements_[0];
      double psi = meas_package.raw_measurements_[1];
      x_ << rho * cos(psi),
          rho * sin(psi),
          0,
          0,
          0;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  auto delta_t = (double) (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    UpdateLidar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    UpdateRadar(meas_package);
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  const MatrixXd A = P_aug.llt().matrixL();

  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  for (size_t i = 0; i < Xsig_aug.cols(); ++i) {
    const VectorXd &x_k = Xsig_aug.col(i);
    VectorXd tmp = VectorXd(n_x_);
    if (x_k[4] == 0) {
      tmp << x_k[2] * cos(x_k[3]) * delta_t + 0.5 * pow(delta_t, 2) * cos(x_k[3]) * x_k[5],
          x_k[2] * sin(x_k[3]) * delta_t + 0.5 * pow(delta_t, 2) * sin(x_k[3]) * x_k[5],
          delta_t * x_k[5],
          x_k[4] * delta_t + 0.5 * pow(delta_t, 2) * x_k[6],
          delta_t * x_k[6];
    } else {
      tmp << x_k[2] / x_k[4] * (sin(x_k[3] + delta_t * x_k[4]) - sin(x_k[3])) +
          0.5 * pow(delta_t, 2) * cos(x_k[3]) * x_k[5],
          x_k[2] / x_k[4] * (-cos(x_k[3] + delta_t * x_k[4]) + cos(x_k[3])) +
              0.5 * pow(delta_t, 2) * sin(x_k[3]) * x_k[5],
          delta_t * x_k[5],
          x_k[4] * delta_t + 0.5 * pow(delta_t, 2) * x_k[6],
          delta_t * x_k[6];
    }
    Xsig_pred_.col(i) = x_k.head(n_x_) + tmp;
  }

  // predict state mean
  VectorXd x_tmp = VectorXd::Zero(n_x_);
  for (size_t i = 0; i < Xsig_pred_.cols(); ++i)
    x_tmp += weights_[i] * Xsig_pred_.col(i);
  x_ = x_tmp;

  // predict state covariance matrix
  MatrixXd P_tmp = MatrixXd::Zero(n_x_, n_x_);
  for (size_t i = 0; i < Xsig_pred_.cols(); ++i) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P_tmp += weights_[i] * x_diff * x_diff.transpose();
  }
  P_ = P_tmp;
}

void UKF::UpdateLidar(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // Measurement
  VectorXd z = meas_package.raw_measurements_;

  // transform sigma points into measurement space
  MatrixXd Zsig = MatrixXd(n_lidar_, 2 * n_aug_ + 1);
  //Zsig = Xsig_pred_.topLeftCorner(2, 2 * n_aug_ + 1);
  for (size_t i = 0; i < Zsig.cols(); ++i){
    const VectorXd& x = Xsig_pred_.col(i);
    Zsig.col(i)[0] = x[0];
    Zsig.col(i)[1] = x[1];
  }

  // calculate mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_lidar_);
  for (size_t i = 0; i < Zsig.cols(); ++i) {
    z_pred += weights_[i] * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_lidar_, n_lidar_);
  for (size_t i = 0; i < Zsig.cols(); ++i) {
    MatrixXd z_diff = Zsig.col(i) - z_pred;
    S += weights_[i] * z_diff * z_diff.transpose();
  }
  S += R_lidar_;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_lidar_);
  for (size_t i = 0; i < Zsig.cols(); ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    Tc += weights_[i] * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

void UKF::UpdateRadar(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // Measurement
  VectorXd z = meas_package.raw_measurements_;

  // transform sigma points into measurement space
  MatrixXd Zsig = MatrixXd(n_radar_, 2 * n_aug_ + 1);
  for (size_t i = 0; i < Zsig.cols(); ++i){
    const VectorXd& x = Xsig_pred_.col(i);
    double rho = x.head(2).norm();
    double psi = atan2(x[1], x[0]);
    double rho_dot = (x[0] * cos(x[3]) * x[2] + x[1] * sin(x[3]) * x[2]) / rho;
    Zsig.col(i)[0] = rho;
    Zsig.col(i)[1] = psi;
    Zsig.col(i)[2] = rho_dot;
  }

  // calculate mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_radar_);
  for (size_t i = 0; i < Zsig.cols(); ++i) {
    z_pred += weights_[i] * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_radar_, n_radar_);
  for (size_t i = 0; i < Zsig.cols(); ++i) {
    MatrixXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_[i] * z_diff * z_diff.transpose();
  }
  S += R_radar_;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_radar_);
  for (size_t i = 0; i < Zsig.cols(); ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    Tc += weights_[i] * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}