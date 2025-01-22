#ifndef TENSOR_H
#define TENSOR_H

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp> 
#include <xtensor/xio.hpp>
#include <vector>
#include <memory>
#include <functional>
#include <set>
#include <string>
#include <algorithm>
#include <iostream>

class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
    /// The actual data of this tensor
    xt::xarray<double> data;
    /// The gradient of this tensor (same shape as data)
    xt::xarray<double> grad;

    /**
     * _backward is a lambda/function that, when called, will
     * apply the chain rule logic for this node to update the .grad
     * of its predecessors.
     */
    std::function<void()> _backward;

    /**
     * The list of previous nodes in the graph that this Tensor
     * depends on; we store them as std::weak_ptr to avoid
     * cyclic references.
     */
    std::vector<std::weak_ptr<Tensor>> _prev;

    /// A string to store the operation that created this Tensor (debug only)
    std::string _op;

public:
    Tensor(const xt::xarray<double>& data)
        : data(data),
          grad(xt::zeros<double>(data.shape())), // same shape, filled with 0
          _backward([](){ /* do nothing by default */ }),
          _op("")
    {
    }

    Tensor(const xt::xarray<double>& data, 
           const std::vector<std::shared_ptr<Tensor>>& children, 
           const std::string& op="")
        : data(data),
          grad(xt::zeros<double>(data.shape())),
          _backward([](){ /* do nothing by default */ }),
          _op(op)
    {
        for (auto& child : children) {
            _prev.push_back(child);
        }
    }


    static std::shared_ptr<Tensor> create(const xt::xarray<double>& data)
    {
        return std::make_shared<Tensor>(data);
    }

    /**
     * Perform backpropagation from *this* node all the way down.
     * 
     * We'll do a topological sort of the graph by building `topo`,
     * then set our grad to 1.0 (the derivative of 'self' wrt 'self'),
     * and apply chain rule in reverse topological order.
     */
    void backward()
    {
        std::vector<std::shared_ptr<Tensor>> topo;
        std::set<Tensor*> visited;

        std::function<void(std::shared_ptr<Tensor>)> build_topo = 
            [&](std::shared_ptr<Tensor> v)
            {
                if (!visited.count(v.get()))
                {
                    visited.insert(v.get());
                    for (auto& child_wptr : v->_prev) {
                        if (auto child = child_wptr.lock()) {
                            build_topo(child);
                        }
                    }
                    topo.push_back(v);
                }
            };

        build_topo(shared_from_this());

        this->grad = xt::ones<double>(this->data.shape());

        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            (*it)->_backward(); 
        }
    }

    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& other)
    {
        auto out_data = this->data + other->data; 
        auto out = std::make_shared<Tensor>(
            out_data,
            std::vector<std::shared_ptr<Tensor>>{ this->shared_from_this(), other },
            "+"
        );

        auto self = this->shared_from_this(); 
        out->_backward = [out, self, other]() {
            self->grad += out->grad;    
            other->grad += out->grad;
        };
        return out;
    }

    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& other)
    {
        auto out_data = this->data * other->data;
        auto out = std::make_shared<Tensor>(
            out_data,
            std::vector<std::shared_ptr<Tensor>>{ this->shared_from_this(), other },
            "*"
        );

        auto self = this->shared_from_this();
        out->_backward = [out, self, other]() {
            self->grad += out->grad * other->data; 
            other->grad += out->grad * self->data;
        };
        return out;
    }


    std::shared_ptr<Tensor> operator-() 
    {
        auto minus_one = Tensor::create(xt::xarray<double>({-1.0}));
        return (*this) * minus_one;
    }


    std::shared_ptr<Tensor> operator+(double scalar)
    {
        auto scalarTensor = Tensor::create(xt::xarray<double>({scalar}));
        return (*this) + scalarTensor;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t)
    {
        os << "Tensor(data=" << t.data << ", grad=" << t.grad << ", op=" << t._op << ")";
        return os;
    }
};


#endif 
