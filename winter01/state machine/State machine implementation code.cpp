#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

// 前向声明
class Context;

// 事件枚举（所有可能触发状态转换的事件）
enum class Event {
    StartCommand,      // 启动命令
    ObstacleDetected,  // 检测到障碍物
    AvoidComplete,     // 避开完成
    ReturnedHome       // 返回原点
};

// 抽象状态基类
class State {
public:
    virtual ~State() = default;

    // 进入状态时调用
    virtual void entry() {}
    // 退出状态时调用
    virtual void exit() {}

    // 处理事件，根据事件和当前状态决定是否转换
    // 参数 context 用于调用转换函数
    virtual void handleEvent(Event event, Context& context) = 0;
};

// 上下文类：持有当前状态，并提供事件处理接口
class Context {
public:
    // 构造函数：初始状态由外部传入（使用 unique_ptr 管理）
    Context(std::unique_ptr<State> initialState)
        : currentState(std::move(initialState)) {
        currentState->entry();  // 进入初始状态
    }

    // 状态转换函数：安全地替换当前状态
    void transitionTo(std::unique_ptr<State> newState) {
        if (currentState) {
            currentState->exit();   // 退出旧状态
        }
        currentState = std::move(newState);
        if (currentState) {
            currentState->entry();  // 进入新状态
        }
    }

    // 对外接口：接收事件并交给当前状态处理
    void handleEvent(Event event) {
        if (currentState) {
            currentState->handleEvent(event, *this);
        }
    }

private:
    std::unique_ptr<State> currentState;   // 当前状态（智能指针自动管理内存）
};

// ---------- 具体状态类（前向声明，因为相互引用）----------
class IdleState;
class ForwardState;
class AvoidState;
class ReturnState;

// 空闲状态
class IdleState : public State {
public:
    void entry() override {
        std::cout << "[Idle] 进入空闲状态，等待命令..." << std::endl;
    }
    void exit() override {
        std::cout << "[Idle] 退出空闲状态" << std::endl;
    }

    void handleEvent(Event event, Context& context) override {
        if (event == Event::StartCommand) {
            // 收到启动命令，转换到前进状态
            std::cout << "[Idle] 收到启动命令，准备前进" << std::endl;
            context.transitionTo(std::make_unique<ForwardState>());
        }
        // 其他事件忽略（可添加日志）
    }
};

// 前进状态
class ForwardState : public State {
public:
    void entry() override {
        std::cout << "[Forward] 进入前进状态，电机启动" << std::endl;
    }
    void exit() override {
        std::cout << "[Forward] 退出前进状态，电机制动" << std::endl;
    }

    void handleEvent(Event event, EventContext& context) override {
        if (event == Event::ObstacleDetected) {
            // 模拟传感器读取距离
            float distance = getDistance();
            std::cout << "[Forward] 检测到障碍物，距离=" << distance << "cm" << std::endl;

            // 守卫条件：距离小于20cm才触发转换
            if (distance < 20.0f) {
                // 转换前执行动作（停止电机），这里可以调用外部函数
                std::cout << "[Forward] 动作：立即停止电机！" << std::endl;
                // 转换到避障状态
                context.transitionTo(std::make_unique<AvoidState>());
            } else {
                std::cout << "[Forward] 距离较远，继续前进" << std::endl;
            }
        }
        // 其他事件忽略
    }

private:
    // 模拟距离传感器（实际项目中会从硬件读取）
    float getDistance() const {
        // 这里简单返回一个变化的值，演示守卫条件
        static float testDist = 25.0f;
        testDist -= 10.0f;      // 每次调用减少10，模拟靠近障碍物
        if (testDist < 5.0f) testDist = 25.0f; // 重置
        return testDist;
    }
};

// 避障状态
class AvoidState : public State {
public:
    void entry() override {
        std::cout << "[Avoid] 进入避障状态，启动避障算法" << std::endl;
    }
    void exit() override {
        std::cout << "[Avoid] 退出避障状态，停止避障" << std::endl;
    }

    void handleEvent(Event event, Context& context) override {
        if (event == Event::AvoidComplete) {
            std::cout << "[Avoid] 避障完成，准备返回原点" << std::endl;
            context.transitionTo(std::make_unique<ReturnState>());
        }
    }
};

// 返回状态
class ReturnState : public State {
public:
    void entry() override {
        std::cout << "[Return] 进入返回状态，启动返回导航" << std::endl;
    }
    void exit() override {
        std::cout << "[Return] 退出返回状态，停止导航" << std::endl;
    }

    void handleEvent(Event event, Context& context) override {
        if (event == Event::ReturnedHome) {
            std::cout << "[Return] 已返回原点" << std::endl;
            context.transitionTo(std::make_unique<IdleState>());
        }
    }
};

// ---------- 主函数：演示状态机运行 ----------
int main() {
    // 创建上下文，初始状态为空闲
    Context robot(std::make_unique<IdleState>());

    // 模拟一系列事件触发
    std::cout << "\n=== 发送启动命令 ===" << std::endl;
    robot.handleEvent(Event::StartCommand);   // 空闲 -> 前进

    std::cout << "\n=== 第一次障碍物检测 ===" << std::endl;
    robot.handleEvent(Event::ObstacleDetected); // 前进可能触发转换

    std::cout << "\n=== 第二次障碍物检测 ===" << std::endl;
    robot.handleEvent(Event::ObstacleDetected); // 再次检测（距离不同）

    std::cout << "\n=== 发送避开完成事件 ===" << std::endl;
    robot.handleEvent(Event::AvoidComplete);    // 避障 -> 返回

    std::cout << "\n=== 发送返回原点事件 ===" << std::endl;
    robot.handleEvent(Event::ReturnedHome);     // 返回 -> 空闲

    std::cout << "\n=== 再次发送启动命令，验证循环 ===" << std::endl;
    robot.handleEvent(Event::StartCommand);     // 空闲 -> 前进

    // 程序结束，所有状态对象自动析构（unique_ptr管理）
    return 0;
}

/*1. 定义事件枚举
将所有可能的外部事件集中定义，便于扩展。

2. 抽象状态基类 State
定义了三个虚函数：entry()、exit()、handleEvent()。
这样每个具体状态只需实现自己的行为。

3. 上下文类 Context
核心职责：持有当前状态，并作为状态转换的中介。
transitionTo() 负责退出旧状态、切换指针、进入新状态。
使用 std::unique_ptr<State> 自动管理状态对象的生命周期，避免内存泄漏。

4. 具体状态类
每个状态继承自 State，并实现：
entry()：进入时打印信息（实际项目中可做硬件初始化）。
exit()：退出时清理资源。
handleEvent()：根据事件决定是否转换。
守卫条件：在 ForwardState 中通过 if (distance < 20.0f) 实现。
转换动作：在判断条件后、调用 transitionTo 前执行（如 停止电机）。

5. 主函数演示
创建机器人 Context，初始状态为 IdleState。
依次发送事件，观察状态变化和动作执行。
运行结果将显示每个状态的 entry/exit 以及转换时的动作。

*/