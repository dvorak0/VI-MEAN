// The MIT License (MIT)
// Copyright (c) 2014 Matthew Klingensmith and Ivan Dryanovski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef INVERSETRUNCATOR_H_
#define INVERSETRUNCATOR_H_

namespace chisel
{

class InverseTruncator : public Truncator
{
  public:
    InverseTruncator() = default;

    InverseTruncator(float scale)
        : scalingFactor(scale)
    {
    }

    virtual ~InverseTruncator()
    {
    }

    float GetTruncationDistance(float reading) const
    {
        float inv_reading = 1.0 / reading;
        return (DEP_SAMPLE / (inv_reading * inv_reading)) * scalingFactor;
    }

  protected:
    const float BASE_LINE = 0.10;
    const float FOCAL = 471.27;
    const int DEP_CNT = 128;
    const float DEP_SAMPLE = 1.0f / (BASE_LINE * FOCAL);
    const float scalingFactor;
};
typedef std::shared_ptr<InverseTruncator> InverseTruncatorPtr;
typedef std::shared_ptr<const InverseTruncator> InverseTruncatorConstPtr;

} // namespace chisel

#endif // INVERSETRUNCATOR_H_
