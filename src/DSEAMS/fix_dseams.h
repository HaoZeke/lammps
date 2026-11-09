/* d-SEAMS LAMMPS fix.
 *
 * This file includes LAMMPS headers and is distributed under GPL-2.0-only,
 * the licence of LAMMPS. libyodaLib stays MIT and is called through
 * seams_insitu.h.
 *
 * The same header is the in-tree package style and the out-of-tree plugin.
 * Build the plugin with -DDSEAMS_LAMMPS_PLUGIN against the headers of the
 * LAMMPS that will load it. LAMMPS_VERSION in those headers must be the
 * version string of that binary. A plugin built for another version is
 * refused by LAMMPS at load time.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifdef FIX_CLASS
// clang-format off
FixStyle(dseams,FixDseams);
// clang-format on
#else

#ifndef LMP_FIX_DSEAMS_H
#define LMP_FIX_DSEAMS_H

#include "fix.h"

#include <string>
#include <vector>

struct seams_insitu_engine;

namespace LAMMPS_NS {

class FixDseams : public Fix {
 public:
  FixDseams(class LAMMPS *, int, char **);
  ~FixDseams() override;
  int setmask() override;
  void init() override;
  void init_list(int id, class NeighList *ptr) override;
  void setup(int) override;
  void end_of_step() override;

 private:
  int host_;
  int every_;
  int depth_;
  long long seen_step_;
  double cutoff_;
  std::string dump_;
  std::string record_;
  seams_insitu_engine *eng_;
  class NeighList *list_;
  std::vector<double> built_x_;
};

} // namespace LAMMPS_NS

#endif
#endif
