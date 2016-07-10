#ifndef _PATH_EVALUATOR_HPP_
#define _PATH_EVALUATOR_HPP_
#include <iostream>
#include <boost/random.hpp>
#include <boost/timer.hpp>
#include <queue>
#include <robot_environment/robot_environment.hpp>
#include "kalman_filter.hpp"
#include <robot_environment/Obstacle.hpp>
#include "fcl/collision_object.h"
#include <path_planner/dynamic_path_planner.hpp>
#include <path_planner/trajectory.hpp>
#include <unistd.h>
#include <memory>
#include <signal.h>

namespace shared
{

template<class RobotType, class OptionsType>
std::shared_ptr<shared::DynamicPathPlanner> makeDynamicPathPlanner2(std::shared_ptr<shared::RobotEnvironment>& robot_environment,
        std::shared_ptr<OptionsType>& options)
{
    //std::shared_ptr<shared::RobotEnvironment> env = robot_environment->clone<RobotType>();
    std::shared_ptr<shared::DynamicPathPlanner> dyn(new shared::DynamicPathPlanner(false));
    dyn->setup(robot_environment, options->dynamicPlanner);
    std::vector<double> goal_area;
    robot_environment->getGoalArea(goal_area);
    std::vector<double> goal_position( {goal_area[0], goal_area[1], goal_area[2]});
    double goal_radius = goal_area[3];
    std::vector<std::vector<double>> goal_states = robot_environment->getGoalStates();
    ompl::base::GoalPtr goal_region =
        shared::makeRobotGoalRegion(dyn->getSpaceInformation(),
                                    robot_environment,
                                    goal_states);

    dyn->setGoal(goal_region);
    dyn->setControlSampler(options->controlSampler);
    dyn->addIntermediateStates(true);
    dyn->setNumControlSamples(options->numControlSamples);
    dyn->setRRTGoalBias(options->rrtGoalBias);
    dyn->setMinMaxControlDuration(options->minMaxControlDuration);
    return dyn;
}

class PathEvaluationResult
{
public:
    PathEvaluationResult() = default;

    PathEvaluationResult(PathEvaluationResult& res) {
        trajectory = res.trajectory;
        path_objective = res.path_objective;
    }

    PathEvaluationResult& operator=(PathEvaluationResult& res) {
        trajectory = res.trajectory;
        path_objective = res.path_objective;
    }

    shared::Trajectory trajectory;

    double path_objective;

    long numPlannedTrajectories;
};

template<class RobotType, class OptionsType>
class PathEvaluator
{
public:
    PathEvaluator(std::shared_ptr<OptionsType>& options):
        options_(options),
        kalman_filter_(),
        C_(),
        D_(),
        step_penalty_(0.0),
        illegal_move_penalty_(0.0),
        terminal_reward_(0.0),
        discount_factor_(0.0),
        num_samples_(1) {

    }

    std::vector<Eigen::MatrixXd> getLinearModelMatricesState(std::shared_ptr<shared::RobotEnvironment>& env,
            const std::vector<double>& state,
            std::vector<double>& control,
            double control_duration) const {
        std::vector<Eigen::MatrixXd> A_B_V_H_W_M_N;
        std::vector<Eigen::MatrixXd> A_B_V_H_W;
        env->getRobot()->getLinearProcessMatrices(state, control, control_duration, A_B_V_H_W);
        A_B_V_H_W_M_N.push_back(A_B_V_H_W[0]);
        A_B_V_H_W_M_N.push_back(A_B_V_H_W[1]);
        A_B_V_H_W_M_N.push_back(A_B_V_H_W[2]);
        A_B_V_H_W_M_N.push_back(A_B_V_H_W[3]);
        A_B_V_H_W_M_N.push_back(A_B_V_H_W[4]);

        Eigen::MatrixXd M;
        env->getRobot()->getStateCovarianceMatrix(M);
        A_B_V_H_W_M_N.push_back(M);

        Eigen::MatrixXd N;
        env->getRobot()->getObservationCovarianceMatrix(N);
        A_B_V_H_W_M_N.push_back(N);

        return A_B_V_H_W_M_N;
    }

    void getLinearModelMatrices(std::shared_ptr<shared::RobotEnvironment>& env,
                                std::vector<std::vector<double>>& state_path,
                                std::vector<std::vector<double>>& control_path,
                                std::vector<double>& control_durations,
                                std::vector<Eigen::MatrixXd>& As,
                                std::vector<Eigen::MatrixXd>& Bs,
                                std::vector<Eigen::MatrixXd>& Vs,
                                std::vector<Eigen::MatrixXd>& Ms,
                                std::vector<Eigen::MatrixXd>& Hs,
                                std::vector<Eigen::MatrixXd>& Ws,
                                std::vector<Eigen::MatrixXd>& Ns) {
        //std::vector<Eigen::MatrixXd> A_B_V_H_W;
        for (size_t i = 0; i < state_path.size(); i++) {
            std::vector<Eigen::MatrixXd> A_B_V_H_W_M_N = getLinearModelMatricesState(env, state_path[i], control_path[i], control_durations[i]);
            As.push_back(A_B_V_H_W_M_N[0]);
            Bs.push_back(A_B_V_H_W_M_N[1]);
            Vs.push_back(A_B_V_H_W_M_N[2]);
            Hs.push_back(A_B_V_H_W_M_N[3]);
            Ws.push_back(A_B_V_H_W_M_N[4]);
            Ms.push_back(A_B_V_H_W_M_N[5]);
            Ns.push_back(A_B_V_H_W_M_N[6]);
        }
    }

    void setC(Eigen::MatrixXd& C) {
        C_ = C;
    }

    void setD(Eigen::MatrixXd& D) {
        D_ = D;
    }

    bool adjustAndEvaluatePath(std::shared_ptr<shared::RobotEnvironment>& env,
                               shared::Trajectory& trajectory,
                               std::vector<double>& x_estimated,
                               Eigen::MatrixXd& P_t,
                               unsigned int& current_step,
                               std::shared_ptr<shared::PathEvaluationResult>& res) {
        std::vector<double> x_estimated_t = x_estimated;
        std::vector<std::vector<double>> xs;
        std::vector<std::vector<double>> us;
        std::vector<double> control_durations;
        for (size_t i = 1; i < trajectory.xs.size(); i++) {
            xs.push_back(trajectory.xs[i]);
            us.push_back(trajectory.us[i]);
            control_durations.push_back(trajectory.control_durations[i]);
        }

        if (xs.size() < 2) {
            // The trajectory is too short to continue
            return false;
        }

        Eigen::MatrixXd P_t_e(P_t);
        std::vector<Eigen::MatrixXd> As;
        std::vector<Eigen::MatrixXd> Bs;
        std::vector<Eigen::MatrixXd> Vs;
        std::vector<Eigen::MatrixXd> Ms;
        std::vector<Eigen::MatrixXd> Hs;
        std::vector<Eigen::MatrixXd> Ws;
        std::vector<Eigen::MatrixXd> Ns;
        getLinearModelMatrices(env,
                               xs,
                               us,
                               control_durations,
                               As,
                               Bs,
                               Vs,
                               Ms,
                               Hs,
                               Ws,
                               Ns);
        std::vector<Eigen::MatrixXd> Ls;
        unsigned int hor = xs.size() - 1;
        if (!kalman_filter_->computeLGains(As, Bs, C_, D_, hor, Ls)) {
            return false;
        }

        shared::Trajectory adjusted_trajectory;
        adjusted_trajectory.xs.push_back(x_estimated_t);
        adjusted_trajectory.zs.push_back(x_estimated_t);

        std::vector<double> x_tilde = utils_kalman::subtractVectors(x_estimated_t, xs[0]);
        for (size_t i = 0; i < xs.size() - 1; i++) {
            std::vector<double> x_predicted = xs[i];
            VectorXd x_e_minus_p(x_predicted.size());
            for (size_t j = 0; j < x_predicted.size(); j++) {
                x_e_minus_p(j) = x_estimated_t[j] - x_predicted[j];
            }

            Eigen::VectorXd us_i = utils_kalman::toEigenVec(us[i]);
            Eigen::VectorXd u = Ls[i] * x_e_minus_p + us_i;
            std::vector<double> u_vec = utils_kalman::toStdVec(u);
            env->getRobot()->enforceControlConstraints(u_vec);
            std::vector<double> control_error;
            for (size_t j = 0; j < u_vec.size(); j++) {
                control_error.push_back(0.0);
            }

            std::vector<double> propagationResult;
            env->getRobot()->propagateState(xs[i],
                                            u_vec,
                                            control_error,
                                            control_durations[i],
                                            options_->simulationStepSize,
                                            propagationResult);
            adjusted_trajectory.xs.push_back(propagationResult);
            adjusted_trajectory.us.push_back(u_vec);
            std::vector<double> z_elem;
            env->getRobot()->transformToObservationSpace(propagationResult, z_elem);
            adjusted_trajectory.zs.push_back(z_elem);

            std::vector<double> u_dash = utils_kalman::subtractVectors(u_vec, us[i]);

            //Kalman prediction and update
            std::vector<double> x_tilde_dash_t;
            //std::vector<double> x_tilde_estimated;
            std::vector<double> z_dash;
            Eigen::MatrixXd P_t_p;
            kalman_filter_->kalmanPredict(x_tilde,
                                          u_dash,
                                          As[i],
                                          Bs[i],
                                          P_t_e,
                                          Vs[i],
                                          Ms[i],
                                          x_tilde_dash_t,
                                          P_t_p);
            for (size_t j = 0; j < x_tilde_dash_t.size(); j++) {
                z_dash.push_back(0.0);
            }

            kalman_filter_->kalmanUpdate(x_tilde_dash_t,
                                         z_dash,
                                         Hs[i],
                                         P_t_p,
                                         Ws[i],
                                         Ns[i],
                                         x_tilde,
                                         P_t_e);
            x_estimated_t = utils_kalman::addVectors(x_tilde, xs[i + 1]);
        }

        std::vector<double> ze;
        for (size_t i = 0; i < trajectory.us[0].size(); i++) {
            ze.push_back(0.0);
        }

        adjusted_trajectory.us.push_back(ze);
        adjusted_trajectory.control_durations = control_durations;
        double objective = evaluatePath(env,
                                        adjusted_trajectory.xs,
                                        adjusted_trajectory.us,
                                        adjusted_trajectory.control_durations,
                                        P_t,
                                        current_step);
        res = std::make_shared<shared::PathEvaluationResult>();
        res->trajectory = adjusted_trajectory;
        res->path_objective = objective;
        return true;
    }

    double evaluatePath(std::shared_ptr<shared::RobotEnvironment>& env,
                        std::vector<std::vector<double>>& state_path,
                        std::vector<std::vector<double>>& control_path,
                        std::vector<double>& control_durations,
                        Eigen::MatrixXd& P_t_in,
                        unsigned int& current_step) {
        Eigen::MatrixXd P_t(P_t_in);
        std::vector<Eigen::MatrixXd> As;
        std::vector<Eigen::MatrixXd> Bs;
        std::vector<Eigen::MatrixXd> Vs;
        std::vector<Eigen::MatrixXd> Ms;
        std::vector<Eigen::MatrixXd> Hs;
        std::vector<Eigen::MatrixXd> Ws;
        std::vector<Eigen::MatrixXd> Ns;

        getLinearModelMatrices(env,
                               state_path,
                               control_path,
                               control_durations,
                               As,
                               Bs,
                               Vs,
                               Ms,
                               Hs,
                               Ws,
                               Ns);
        double path_reward = 0.0;
        unsigned int horizon_L = state_path.size();
        std::vector<Eigen::MatrixXd> Ls;
        unsigned int hor = horizon_L - 1;
        if (!kalman_filter_->computeLGains(As, Bs, C_, D_, hor, Ls)) {
            return 0.0;
        }
        Eigen::MatrixXd Q_t_up(Ms[0].rows(), Ms[0].cols() + Ns[0].cols());
        Eigen::MatrixXd Q_t_down(Ns[0].rows(), Ms[0].cols() + Ns[0].cols());
        Eigen::MatrixXd Q_t_up_right = Eigen::MatrixXd::Zero(Ms[0].rows(), Ns[0].cols());
        Eigen::MatrixXd Q_t_down_left = Eigen::MatrixXd::Zero(Ns[0].rows(), Ms[0].cols());
        Q_t_up << Ms[0], Q_t_up_right;
        Q_t_down << Q_t_down_left, Ns[0];
        Eigen::MatrixXd Q_t(Q_t_up.rows() + Q_t_down.rows(), Q_t_up.cols());
        Q_t << Q_t_up,
            Q_t_down;
        Eigen::MatrixXd R_t_up_right = Eigen::MatrixXd::Zero(P_t.rows(), P_t.cols());
        Eigen::MatrixXd R_t_up(P_t.rows(), 2 * P_t.cols());
        Eigen::MatrixXd R_t_down(P_t.rows(), 2 * P_t.cols());
        R_t_up << P_t, R_t_up_right;
        R_t_down << R_t_up_right, P_t;
        Eigen::MatrixXd R_t(2 * P_t.rows(), 2 * P_t.cols());
        R_t << R_t_up,
            R_t_down;
        for (size_t i = 1; i < horizon_L; i++) {
            Eigen::MatrixXd P_hat_t;
            kalman_filter_->computePredictedCovariance(As[i], P_t, Vs[i], Ms[i], P_hat_t);
            Eigen::MatrixXd K_t;
            kalman_filter_->computeKalmanGain(Hs[i], P_hat_t, Ws[i], Ns[i], K_t);
            kalman_filter_->computeEstimatedCovariance(K_t, Hs[i], P_hat_t, P_t);
            Eigen::MatrixXd BL(Bs[i] * Ls[i - 1]);
            Eigen::MatrixXd KHA(K_t * Hs[i] * As[i]);
            Eigen::MatrixXd F_l_r(As[i] + BL - KHA);
            Eigen::MatrixXd F_t(As[i].rows() + KHA.rows(), As[i].cols() + BL.cols());
            F_t << As[i], BL,
                KHA, F_l_r;
            Eigen::MatrixXd KHV(K_t * Hs[i] * Vs[i]);
            Eigen::MatrixXd KW(K_t * Ws[i]);
            Eigen::MatrixXd G_u_r = MatrixXd::Zero(Vs[i].rows(), KW.cols());
            Eigen::MatrixXd G_t(Vs[i].rows() + KHV.rows(), Vs[i].cols() + G_u_r.cols());
            G_t << Vs[i], G_u_r,
                KHV, KW;
            R_t = F_t * R_t * F_t.transpose() + G_t * Q_t * G_t.transpose();
            Eigen::MatrixXd Gamma_t_u_l = Eigen::MatrixXd::Identity(Ls[i - 1].rows(), Ls[i - 1].cols());
            Eigen::MatrixXd Gamma_t_u_r = Eigen::MatrixXd::Zero(Gamma_t_u_l.rows(), Ls[i - 1].cols());
            Eigen::MatrixXd Gamma_t_l_l = Eigen::MatrixXd::Zero(Ls[i - 1].rows(), Gamma_t_u_l.cols());
            Eigen::MatrixXd Gamma_t(Gamma_t_u_l.rows() + Gamma_t_l_l.rows(), Gamma_t_u_l.cols() + Gamma_t_u_r.cols());
            Gamma_t << Gamma_t_u_l, Gamma_t_u_r,
                    Gamma_t_l_l, Ls[i - 1];

            Eigen::MatrixXd Cov(Gamma_t * R_t * Gamma_t.transpose());
            Eigen::MatrixXd cov_state(Cov.block(0, 0, P_t.rows(), P_t.cols()));
            double expected_state_reward = 0.0;            
            getExpectedStateReward(env,
                                   state_path[i],
                                   cov_state,
                                   expected_state_reward);
            path_reward += std::pow(discount_factor_, current_step + i) * expected_state_reward;
        }

        return path_reward;
    }

    void setRewardModel(double& step_penalty,
                        double& illegal_move_penalty,
                        double& terminal_reward,
                        double& discount_factor) {
        step_penalty_ = step_penalty;
        illegal_move_penalty_ = illegal_move_penalty;
        terminal_reward_ = terminal_reward;
        discount_factor_ = discount_factor;
    }

    double setNumSamples(unsigned int& num_samples) {
        num_samples_ = num_samples;
    }

    bool planAndEvaluatePaths(const std::vector<double>& start_state,
                              Eigen::MatrixXd& P_t,
                              unsigned int& current_step,
                              unsigned int& num_threads,
                              std::vector<std::shared_ptr<shared::RobotEnvironment>>& robot_environments,
                              std::vector<std::shared_ptr<shared::DynamicPathPlanner>>& dynamic_path_planners,
                              std::shared_ptr<shared::PathEvaluationResult>& res,
                              unsigned int minNumPaths = 0,
                              double timeout = -1) {
        cout << "HFR: Planning with timeout: " << timeout << endl;
        std::shared_ptr<std::queue<std::shared_ptr<shared::PathEvaluationResult>>> queue_ptr(new std::queue<std::shared_ptr<shared::PathEvaluationResult>>);
        std::vector<boost::thread*> threads;
        for (size_t i = 0; i < num_threads; i++) {
            threads.push_back(new boost::thread(&PathEvaluator::eval_thread,
                                                this,
                                                queue_ptr,
                                                robot_environments[i],
                                                dynamic_path_planners[i],
                                                start_state,
                                                P_t,
                                                current_step));
        }

        if (timeout < 0) {
            usleep(options_->stepTimeout * 1000.0);
        } else {
            usleep(timeout * 1000.0);
        }

        if (minNumPaths > 0) {
            while (queue_ptr->size() < minNumPaths) {
                usleep(10);
            }
        }
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i]->interrupt();
        }

        for (size_t i = 0; i < threads.size(); i++) {
            threads[i]->join();
        }

        for (size_t i = 0; i < threads.size(); i++) {
            delete threads[i];
        }

        threads.clear();
        double best_objective = -1000000;
        unsigned int queue_size = queue_ptr->size();
        for (size_t i = 0; i < queue_size; i++) {
            std::shared_ptr<shared::PathEvaluationResult> next_queue_elem = queue_ptr->front();
            if (next_queue_elem->path_objective > best_objective) {
                best_objective = next_queue_elem->path_objective;
                res = std::make_shared<shared::PathEvaluationResult>(*(next_queue_elem.get()));
                res->path_objective = next_queue_elem->path_objective;

            }

            queue_ptr->pop();
        }

        if (res) {
            res->numPlannedTrajectories = queue_size;
        }

    }

    bool eval_thread(std::shared_ptr<std::queue<std::shared_ptr<shared::PathEvaluationResult>>>& queue_ptr,
                     std::shared_ptr<shared::RobotEnvironment>& env,
                     std::shared_ptr<shared::DynamicPathPlanner>& dynamic_path_planner,
                     const std::vector<double>& start_state,
                     Eigen::MatrixXd& P_t,
                     unsigned int& current_step) {
        while (true) {
            try {
                dynamic_path_planner->reset();
                std::vector<std::vector<double>> solution = dynamic_path_planner->solve(start_state,
                                              options_->rrtTimeout / 1000.0);

                if (solution.size() != 0) {
                    unsigned int state_space_dimension = env->getRobot()->getStateSpaceDimension();
                    unsigned int control_space_dimension = env->getRobot()->getControlSpaceDimension();
                    std::vector<std::vector<double>> xs;
                    std::vector<std::vector<double>> us;
                    std::vector<std::vector<double>> zs;
                    std::vector<double> control_durations;
                    for (size_t i = 0; i < solution.size(); i++) {
                        std::vector<double> x_elem;
                        std::vector<double> u_elem;
                        std::vector<double> z_elem;
                        for (size_t j = 0; j < state_space_dimension; j++) {
                            x_elem.push_back(solution[i][j]);
                            if (j < control_space_dimension) {
                                u_elem.push_back(solution[i][state_space_dimension + j]);
                            }
                        }
                        env->getRobot()->transformToObservationSpace(x_elem, z_elem);


                        control_durations.push_back(solution[i][2 * state_space_dimension + control_space_dimension]);
                        xs.push_back(x_elem);
                        us.push_back(u_elem);
                        zs.push_back(z_elem);
                    }

                    //Evaluate the solution
                    boost::this_thread::interruption_point();
                    double objective = evaluatePath(env, xs, us, control_durations, P_t, current_step);
                    shared::Trajectory trajectory;
                    trajectory.xs = xs;
                    trajectory.us = us;
                    trajectory.zs = zs;
                    trajectory.control_durations = control_durations;
                    std::shared_ptr<PathEvaluationResult> result(new PathEvaluationResult());
                    result->trajectory = trajectory;
                    result->path_objective = objective;
                    boost::this_thread::interruption_point();
                    mtx_.lock();
                    queue_ptr->push(result);
                    mtx_.unlock();
                    boost::this_thread::interruption_point();
                }
            }

            catch (boost::thread_interrupted&) {
                dynamic_path_planner->reset();
                return true;
            }
        }
    }

    std::shared_ptr<shared::KalmanFilter> getKalmanFilter() const {
        return kalman_filter_;
    }

private:
    std::shared_ptr<OptionsType> options_;

    boost::mutex mtx_;

    std::shared_ptr<shared::KalmanFilter> kalman_filter_;

    Eigen::MatrixXd C_;
    Eigen::MatrixXd D_;

    double step_penalty_;
    double illegal_move_penalty_;
    double terminal_reward_;
    double discount_factor_;

    unsigned int num_samples_;

    bool getExpectedStateReward(std::shared_ptr<shared::RobotEnvironment>& env,
                                std::vector<double>& state,
                                Eigen::MatrixXd& cov_state,
                                double& expected_state_reward) {
        std::vector<std::vector<double>> state_samples;
        if (!sampleValidStates(env, state, cov_state, num_samples_, state_samples)) {
            cout << "RETURN FALSE" << endl;
            return false;
        }

        bool collides = false;
        std::vector<std::shared_ptr<shared::Obstacle>> obstacles;
        env->getObstacles(obstacles);
        for (size_t i = 0; i < num_samples_; i++) {
            // Check for collision
            std::vector<std::shared_ptr<fcl::CollisionObject>> collision_objects;
            env->getRobot()->createRobotCollisionObjects(state_samples[i], collision_objects);
            collides = false;
            for (size_t j = 0; j < obstacles.size(); j++) {
                if (!obstacles[j]->getTerrain()->isTraversable()) {
                    if (obstacles[j]->in_collision(collision_objects)) {
                        expected_state_reward -= illegal_move_penalty_;
                        collides = true;
                        break;
                    }
                }
            }

            if (!collides) {
                if (env->getRobot()->isTerminal(state_samples[i])) {
                    expected_state_reward += terminal_reward_;
                }

                else {
                    expected_state_reward -= step_penalty_;
                }
            }

        }
        expected_state_reward /= float(num_samples_);
        return true;
    }

    bool sampleValidStates(std::shared_ptr<shared::RobotEnvironment>& env,
                           std::vector<double>& mean,
                           Eigen::MatrixXd& cov,
                           unsigned int num_samples,
                           std::vector<std::vector<double>>& samples) {
        samples.clear();
        Eigen::MatrixXd mean_matr(mean.size(), 1);
        for (size_t i = 0; i < mean.size(); i++) {
            mean_matr(i) = mean[i];
        }

        std::shared_ptr<shared::Robot> robot = env->getRobot();
        unsigned int seed = std::time(nullptr);
        std::shared_ptr<Eigen::EigenMultivariateNormal<double>> distr = env->createDistribution(mean_matr, cov, seed);
        Eigen::MatrixXd samples_e = distr->samples(num_samples);        
        for (size_t i = 0; i < num_samples; i++) {
            std::vector<double> sample;
            for (size_t j = 0; j < mean.size(); j++) {
                sample.push_back(samples_e(j, i));
            }

            if (robot->constraintsEnforced()) {
                robot->enforceConstraints(sample);
            }

            samples.push_back(sample);
        }

        return true;
    }

};

}

#endif
