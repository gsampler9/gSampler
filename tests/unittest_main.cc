#include <gtest/gtest.h>
#include <torch/torch.h>

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
