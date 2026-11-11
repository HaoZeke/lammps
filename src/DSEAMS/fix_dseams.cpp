// Copyright (c) d-SEAMS developers
/* d-SEAMS LAMMPS fix.
 *
 * This file includes LAMMPS headers and is distributed under GPL-2.0-only,
 * the licence of LAMMPS. libyodaLib stays MIT and is called through
 * seams_insitu.h.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "fix_dseams.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "lammpsplugin.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "version.h"

#include <seams_insitu.h>

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

namespace {

int parse_positive_int(const char *text, int *out)
{
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 1 || value > 1000000) { return 1; }
  *out = static_cast<int>(value);
  return 0;
}

double parse_positive_double(const char *text, int *bad)
{
  char *end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || !(value > 0.0)) {
    *bad = 1;
    return 0;
  }
  return value;
}

}    // namespace

FixDseams::FixDseams(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), host_(1), every_(1), depth_(8), neigh_open_(0), seen_step_(-1),
    cutoff_(3.5), eng_(nullptr), list_(nullptr)
{
  int iarg = 3;
  while (iarg < narg) {
    if (std::strcmp(arg[iarg], "dump") == 0 && iarg + 1 < narg) {
      dump_ = arg[iarg + 1];
      iarg += 2;
    } else if (std::strcmp(arg[iarg], "record") == 0 && iarg + 1 < narg) {
      record_ = arg[iarg + 1];
      iarg += 2;
    } else if (std::strcmp(arg[iarg], "every") == 0 && iarg + 1 < narg) {
      if (parse_positive_int(arg[iarg + 1], &every_) != 0) {
        error->all(FLERR, "fix dseams: every must be a positive integer");
      }
      iarg += 2;
    } else if (std::strcmp(arg[iarg], "cutoff") == 0 && iarg + 1 < narg) {
      int bad = 0;
      cutoff_ = parse_positive_double(arg[iarg + 1], &bad);
      if (bad) { error->all(FLERR, "fix dseams: cutoff must be positive"); }
      iarg += 2;
    } else if (std::strcmp(arg[iarg], "depth") == 0 && iarg + 1 < narg) {
      if (parse_positive_int(arg[iarg + 1], &depth_) != 0) {
        error->all(FLERR, "fix dseams: depth must be a positive integer");
      }
      iarg += 2;
    } else if (std::strcmp(arg[iarg], "neigh") == 0 && iarg + 1 < narg) {
      neigh_ = arg[iarg + 1];
      iarg += 2;
    } else if (std::strcmp(arg[iarg], "host") == 0 && iarg + 1 < narg) {
      if (std::strcmp(arg[iarg + 1], "yes") == 0) {
        host_ = 1;
      } else if (std::strcmp(arg[iarg + 1], "no") == 0) {
        host_ = 0;
      } else {
        error->all(FLERR, "fix dseams: host is yes or no");
      }
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix dseams command");
    }
  }
  if (record_.empty()) { error->all(FLERR, "fix dseams: record is required"); }
  int opened = 0;
  if (comm->me == 0) {
    seams_insitu_options opt;
    seams_insitu_options_default(&opt);
    opt.cutoff = cutoff_;
    opt.ring_depth = depth_;
    opt.cadence.every = 1;
    opt.debug_dump = dump_.empty() ? 0 : 1;
    eng_ = seams_insitu_create(&opt);
    // depth outside 3..12 makes create return null. Do not open that pointer.
    if (eng_ != nullptr && seams_insitu_open_record(eng_, record_.c_str()) == SEAMS_INSITU_OK) {
      opened = 1;
    }
  }
  MPI_Bcast(&opened, 1, MPI_INT, 0, lmp->world);
  if (opened == 0) { error->all(FLERR, "fix dseams: record did not open"); }
}

FixDseams::~FixDseams()
{
  seams_insitu_destroy(eng_);
  eng_ = nullptr;
}

int FixDseams::setmask()
{
  int mask = 0;
  mask |= FixConst::END_OF_STEP;
  return mask;
}

void FixDseams::init()
{
  list_ = nullptr;
  // stable_30Sep2026 dropped set_cutoff(). An analysis cutoff is the same
  // for every type pair, which is set_cutoff_fixed.
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL)
      ->set_cutoff_fixed(cutoff_);
}

void FixDseams::init_list(int, NeighList *ptr)
{
  list_ = ptr;
}

// LAMMPS dump writes the initial frame at timestep 0. end_of_step does not
// run until a step has finished, so the census missed that frame.
void FixDseams::setup(int)
{
  end_of_step();
}

void FixDseams::end_of_step()
{
  const auto step64 = static_cast<long long>(update->ntimestep);
  if ((step64 % every_) != 0) { return; }
  // setup runs at the first step of every run. A later run starts on the
  // step the previous run already analysed.
  if (step64 == seen_step_) { return; }
  seen_step_ = step64;
  const int nlocal = atom->nlocal;
  std::vector<double> local(static_cast<std::size_t>(nlocal) * 3);
  std::vector<int> local_types(static_cast<std::size_t>(nlocal));
  std::vector<int> local_ids(static_cast<std::size_t>(nlocal));
  std::vector<int> mask(static_cast<std::size_t>(nlocal));
  for (int i = 0; i < nlocal; ++i) {
    local[static_cast<std::size_t>(3 * i)] = atom->x[i][0];
    local[static_cast<std::size_t>(3 * i + 1)] = atom->x[i][1];
    local[static_cast<std::size_t>(3 * i + 2)] = atom->x[i][2];
    local_types[static_cast<std::size_t>(i)] = atom->type[i];
    local_ids[static_cast<std::size_t>(i)] = static_cast<int>(atom->tag[i]);
    mask[static_cast<std::size_t>(i)] = atom->mask[i];
  }
  std::vector<double> kept_x(local.size());
  std::vector<int> kept_t(static_cast<std::size_t>(nlocal));
  std::vector<int> kept_i(static_cast<std::size_t>(nlocal));
  const int kept =
      seams_insitu_pack_owned(nlocal, mask.data(), groupbit, local.data(), local_types.data(),
                              local_ids.data(), kept_x.data(), kept_t.data(), kept_i.data());
  const int nprocs = comm->nprocs;
  std::vector<int> counts(static_cast<std::size_t>(nprocs));
  std::vector<int> displs(static_cast<std::size_t>(nprocs));
  int nkept = kept;
  MPI_Allgather(&nkept, 1, MPI_INT, counts.data(), 1, MPI_INT, lmp->world);
  int total = 0;
  seams_insitu_gather_displs(nprocs, counts.data(), displs.data(), &total);
  std::vector<int> xcounts(static_cast<std::size_t>(nprocs));
  std::vector<int> xdispls(static_cast<std::size_t>(nprocs));
  for (int rank = 0; rank < nprocs; ++rank) {
    xcounts[static_cast<std::size_t>(rank)] = counts[static_cast<std::size_t>(rank)] * 3;
    xdispls[static_cast<std::size_t>(rank)] = displs[static_cast<std::size_t>(rank)] * 3;
  }
  std::vector<double> all_x(static_cast<std::size_t>(total) * 3);
  std::vector<int> all_t(static_cast<std::size_t>(total));
  std::vector<int> all_i(static_cast<std::size_t>(total));
  double xpad = 0.0;
  int ipad = 0;
  MPI_Gatherv(kept > 0 ? kept_x.data() : &xpad, kept * 3, MPI_DOUBLE,
              all_x.empty() ? &xpad : all_x.data(), xcounts.data(), xdispls.data(), MPI_DOUBLE, 0,
              lmp->world);
  MPI_Gatherv(kept > 0 ? kept_t.data() : &ipad, kept, MPI_INT, all_t.empty() ? &ipad : all_t.data(),
              counts.data(), displs.data(), MPI_INT, 0, lmp->world);
  MPI_Gatherv(kept > 0 ? kept_i.data() : &ipad, kept, MPI_INT, all_i.empty() ? &ipad : all_i.data(),
              counts.data(), displs.data(), MPI_INT, 0, lmp->world);
  // Every rank contributes its full neighbour rows. Ghost images keep the
  // real atom tag, and the engine drops repeated images inside the cutoff.
  // host no is the seam a coverage input uses to skip that list. The
  // default keeps it. The environment is not read.
  int want = (list_ != nullptr && host_ != 0) ? 1 : 0;
  int local_stale = want;
  if (want && built_x_.size() == static_cast<std::size_t>(kept) * 3) {
    local_stale = 0;
    constexpr double half = 0.25;
    for (int i = 0; i < kept; ++i) {
      const double dx =
          kept_x[static_cast<std::size_t>(3 * i)] - built_x_[static_cast<std::size_t>(3 * i)];
      const double dy = kept_x[static_cast<std::size_t>(3 * i + 1)] -
          built_x_[static_cast<std::size_t>(3 * i + 1)];
      const double dz = kept_x[static_cast<std::size_t>(3 * i + 2)] -
          built_x_[static_cast<std::size_t>(3 * i + 2)];
      if (dx * dx + dy * dy + dz * dz > half * half) {
        local_stale = 1;
        break;
      }
    }
  }
  int stale = 0;
  MPI_Allreduce(&local_stale, &stale, 1, MPI_INT, MPI_MAX, lmp->world);
  std::vector<int> nrow_counts(static_cast<std::size_t>(nprocs));
  std::vector<int> nrow_displs(static_cast<std::size_t>(nprocs));
  std::vector<int> nflat_counts(static_cast<std::size_t>(nprocs));
  std::vector<int> nflat_displs(static_cast<std::size_t>(nprocs));
  std::vector<int> all_counts;
  std::vector<int> all_flat;
  int nrows = 0;
  int nflat = 0;
  if (stale) {
    std::vector<int> row_counts;
    std::vector<int> flat;
    neighbor->build_one(list_);
    for (int ii = 0; ii < list_->inum; ++ii) {
      const int i = list_->ilist[ii];
      if ((atom->mask[i] & groupbit) == 0) { continue; }
      const int begin = static_cast<int>(flat.size());
      flat.push_back(static_cast<int>(atom->tag[i]));
      int *jlist = list_->firstneigh[i];
      for (int jj = 0; jj < list_->numneigh[i]; ++jj) {
        const int j = jlist[jj] & NEIGHMASK;
        if ((atom->mask[j] & groupbit) == 0) { continue; }
        flat.push_back(static_cast<int>(atom->tag[j]));
      }
      row_counts.push_back(static_cast<int>(flat.size()) - begin);
    }
    int local_rows = static_cast<int>(row_counts.size());
    int local_flat = static_cast<int>(flat.size());
    MPI_Allgather(&local_rows, 1, MPI_INT, nrow_counts.data(), 1, MPI_INT, lmp->world);
    MPI_Allgather(&local_flat, 1, MPI_INT, nflat_counts.data(), 1, MPI_INT, lmp->world);
    seams_insitu_gather_displs(nprocs, nrow_counts.data(), nrow_displs.data(), &nrows);
    seams_insitu_gather_displs(nprocs, nflat_counts.data(), nflat_displs.data(), &nflat);
    all_counts.resize(static_cast<std::size_t>(std::max(nrows, 1)));
    all_flat.resize(static_cast<std::size_t>(std::max(nflat, 1)));
    MPI_Gatherv(row_counts.data(), local_rows, MPI_INT, all_counts.data(), nrow_counts.data(),
                nrow_displs.data(), MPI_INT, 0, lmp->world);
    MPI_Gatherv(flat.data(), local_flat, MPI_INT, all_flat.data(), nflat_counts.data(),
                nflat_displs.data(), MPI_INT, 0, lmp->world);
    built_x_.assign(kept_x.begin(), kept_x.begin() + static_cast<std::ptrdiff_t>(kept) * 3);
    if (comm->me == 0) {
      host_counts_.assign(all_counts.begin(), all_counts.begin() + nrows);
      host_flat_.assign(all_flat.begin(), all_flat.begin() + nflat);
    }
  }
  if (comm->me != 0) { return; }
  std::vector<int> order(static_cast<std::size_t>(total));
  for (int i = 0; i < total; ++i) { order[static_cast<std::size_t>(i)] = i; }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return all_i[static_cast<std::size_t>(a)] < all_i[static_cast<std::size_t>(b)];
  });
  std::vector<double> sorted_x(all_x.size());
  std::vector<int> sorted_t(all_t.size());
  std::vector<int> sorted_i(all_i.size());
  for (int i = 0; i < total; ++i) {
    const int src = order[static_cast<std::size_t>(i)];
    sorted_x[static_cast<std::size_t>(3 * i)] = all_x[static_cast<std::size_t>(3 * src)];
    sorted_x[static_cast<std::size_t>(3 * i + 1)] = all_x[static_cast<std::size_t>(3 * src + 1)];
    sorted_x[static_cast<std::size_t>(3 * i + 2)] = all_x[static_cast<std::size_t>(3 * src + 2)];
    sorted_t[static_cast<std::size_t>(i)] = all_t[static_cast<std::size_t>(src)];
    sorted_i[static_cast<std::size_t>(i)] = all_i[static_cast<std::size_t>(src)];
  }
  seams_insitu_frame frame{};
  frame.step = static_cast<int>(step64);
  frame.n = total;
  frame.xyz = sorted_x.data();
  frame.types = sorted_t.data();
  frame.ids = sorted_i.data();
  frame.xlo = domain->boxlo[0];
  frame.ylo = domain->boxlo[1];
  frame.zlo = domain->boxlo[2];
  frame.lx = domain->prd[0];
  frame.ly = domain->prd[1];
  frame.lz = domain->prd[2];
  frame.xy = domain->xy;
  frame.xz = domain->xz;
  frame.yz = domain->yz;
  frame.gathered = 1;
  frame.local_only = 0;
  frame.rank = 0;
  frame.nranks = nprocs;
  if (stale) { seams_insitu_set_neighbours(eng_, nrows, all_counts.data(), all_flat.data()); }
  if (!neigh_.empty()) {
    write_host(static_cast<int>(step64), total, sorted_x.data(), sorted_t.data(), sorted_i.data());
  }
  const char *dump = dump_.empty() ? nullptr : dump_.c_str();
  const int rc = seams_insitu_observe(eng_, &frame, dump);
  (void) rc;
}

void FixDseams::write_host(int step, int n, const double *xyz, const int *types, const int *ids)
{
  std::ofstream out(neigh_, neigh_open_ ? std::ios::app : std::ios::out);
  neigh_open_ = 1;
  out << "frame\n";
  out << "step " << step << "\n";
  char cutoff_line[64];
  std::snprintf(cutoff_line, sizeof(cutoff_line), "cutoff %.17g\n", cutoff_);
  out << cutoff_line;
  char box_line[320];
  std::snprintf(box_line, sizeof(box_line),
                "box %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", domain->boxlo[0],
                domain->boxlo[1], domain->boxlo[2], domain->prd[0], domain->prd[1], domain->prd[2],
                domain->xy, domain->xz, domain->yz);
  out << box_line;
  out << "pbc " << domain->boundary[0][0] << " " << domain->boundary[1][0] << " "
      << domain->boundary[2][0] << "\n";
  out << "host " << (host_counts_.empty() ? 0 : 1) << "\n";
  out << "atoms " << n << "\n";
  for (int i = 0; i < n; ++i) {
    char line[160];
    std::snprintf(line, sizeof(line), "atom %d %d %.17g %.17g %.17g\n", ids[i], types[i],
                  xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    out << line;
  }
  int flat_at = 0;
  for (int count : host_counts_) {
    out << "row";
    for (int k = 0; k < count; ++k) {
      out << " " << host_flat_[static_cast<std::size_t>(flat_at + k)];
    }
    out << "\n";
    flat_at += count;
  }
  out << "endframe\n";
}

#ifdef DSEAMS_LAMMPS_PLUGIN
static Fix *dseams_creator(LAMMPS *lmp, int narg, char **arg)
{
  return new FixDseams(lmp, narg, arg);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  auto *register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "dseams";
  plugin.info = "d-SEAMS in-situ ring census, CHILL+ and cages";
  plugin.author = "Rohit Goswami (rgoswami@ieee.org)";
  plugin.creator.v2 = reinterpret_cast<lammpsplugin_factory2 *>(&dseams_creator);
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
#endif
