import numpy as np
import pandas as pd
from functools import lru_cache
from itertools import combinations
from typing import List, FrozenSet, Union
from loguru import logger
from typing import Tuple
from algorithms.squeeze.attribute_combination import AttributeCombination as AC
from algorithms.squeeze.anomaly_amount_fileter import KPIFilter
from algorithms.squeeze.squeeze_option import SqueezeOption
from algorithms.squeeze.clustering import cluster_factory
from scipy.spatial.distance import cityblock
from copy import deepcopy


class Squeeze:
    def __init__(self, data_list: List[pd.DataFrame], op=lambda x: x, option: SqueezeOption = SqueezeOption()):
        """
        :param data_list: dataframe without index,
            must have 'real' and 'predict' columns, other columns are considered as attributes
            all elements in this list must have exactly the same attribute combinations in the same order
        """
        self.option = option

        self.one_dim_cluster = cluster_factory(self.option)
        self.cluster_list = []  # type: List[np.ndarray]

        valid_idx = np.logical_and.reduce(
            [_.predict > 0 for _ in data_list],
        )

        self.data_list = list(_[valid_idx] for _ in data_list)
        self.op = op
        self.derived_data = self.get_derived_dataframe(None)  # type: pd.DataFrame
        # There is an error in injection
        self.derived_data.real -= min(np.min(self.derived_data.real), 0)

        self.attribute_names = list(sorted(set(self.derived_data.columns) - {'real', 'predict'}))
        logger.debug(f"available attributes: {self.attribute_names}")

        self.derived_data.sort_values(by=self.attribute_names, inplace=True)
        self.data_list = list(map(lambda x: x.sort_values(by=self.attribute_names), self.data_list))

        self.attribute_values = list(list(set(self.derived_data.loc[:, name].values)) for name in self.attribute_names)
        logger.debug(f"available values: {self.attribute_values}")

        self.ac_array = np.asarray(
            [AC(**record) for record in self.derived_data[self.attribute_names].to_dict(orient='records')])

        self._v = self.derived_data['real'].values
        self._f = self.derived_data['predict'].values
        assert all(self._v >= 0) and all(self._f >= 0), \
            f"currently we assume that KPIs are non-negative, {self.derived_data[~(self._f >= 0)]}"

        self.__finished = False
        self._root_cause = []

        self.filtered_indices = None

    @property
    @lru_cache()
    def root_cause(self):
        return self._root_cause

    @property
    @lru_cache()
    def root_cause_string_list(self):
        unique_root_cause = np.unique(self.root_cause)
        root_cause = list(AC.batch_to_string(_) for _ in unique_root_cause)
        return root_cause

    @property
    @lru_cache()
    def report(self) -> str:
        cluster_impacts = [
            np.sum(np.abs(self._f[idx] - self._v[idx])) for idx in self.cluster_list
        ]
        unique_root_cause, rc_indies = np.unique(self.root_cause, return_index=True)
        cluster_impacts = [
            np.sum(cluster_impacts[idx]) for idx in rc_indies
        ]
        logger.debug(f"{unique_root_cause}, {cluster_impacts}")
        report_df = pd.DataFrame(columns=['root_cause', 'impact'])
        report_df['root_cause'] = list(AC.batch_to_string(_) for _ in unique_root_cause)
        report_df['impact'] = cluster_impacts
        report_df.sort_values(by=['impact'], inplace=True, ascending=False)
        return report_df.to_csv(index=False)

    @lru_cache()
    def get_cuboid_ac_array(self, cuboid: Tuple[str, ...]):
        return np.asarray(list(map(lambda x: x.mask(cuboid), self.ac_array)))

    @lru_cache()
    def get_indexed_data(self, cuboid: Tuple[str, ...]):
        return self.derived_data.set_index(list(cuboid))

    @property
    @lru_cache()
    def normal_indices(self):
        abnormal = np.sort(np.concatenate(self.cluster_list))
        idx = np.argsort(np.abs(self.leaf_deviation_score[abnormal]))
        abnormal = abnormal[idx]
        normal = np.where(np.abs(self.leaf_deviation_score) < self.leaf_deviation_score[abnormal[0]])[0]
        # normal = np.setdiff1d(np.arange(len(self.derived_data)), abnormal, assume_unique=True)
        # return np.intersect1d(normal, self.filtered_indices, assume_unique=True)
        return normal

    def run(self):
        if self.__finished:
            logger.warning(f"try to rerun {self}")
            return self
        if self.option.enable_filter:
            kpi_filter = KPIFilter(self._v, self._f)
            self.filtered_indices = kpi_filter.filtered_indices
            cluster_list = self.one_dim_cluster(self.leaf_deviation_score[self.filtered_indices])
            cluster_list = list(
                [kpi_filter.inverse_map(_) for _ in cluster_list]
            )
            cluster_list = list(
                [list(
                    filter(lambda x: np.min(self.leaf_deviation_score[_]) <= self.leaf_deviation_score[x] <= np.max(
                        self.leaf_deviation_score[_]), np.arange(len(self._f)))
                )
                    for _ in cluster_list]
            )
            self.cluster_list = cluster_list
        else:
            self.filtered_indices = np.ones(len(self._v), dtype=bool)
            self.cluster_list = self.one_dim_cluster(self.leaf_deviation_score)

        self.locate_root_cause()
        self.__finished = True
        self._root_cause = self._root_cause
        return self

    def _locate_in_cuboid(self, cuboid, indices, **params) -> Tuple[FrozenSet[AC], float]:
        """
        :param cuboid: try to find root cause in this cuboid
        :param indices: anomaly leaf nodes' indices
        :return: root causes and their score
        """
        # mu = params.get("mu")
        # sigma = params.get("sigma")
        data_cuboid_indexed = self.get_indexed_data(cuboid)
        logger.debug(f"current cuboid: {cuboid}")

        abnormal_cuboid_ac_arr = self.get_cuboid_ac_array(cuboid)[indices]
        elements, num_elements = np.unique(abnormal_cuboid_ac_arr, return_counts=True)

        num_ele_descents = np.asarray(list(
            np.count_nonzero(
                _.index_dataframe(data_cuboid_indexed),
            ) for _ in elements
        ))

        # sort reversely by descent score
        descent_score = num_elements / np.maximum(num_ele_descents, 1e-4)
        idx = np.argsort(descent_score)[::-1]
        elements = elements[idx]
        num_ele_descents = num_ele_descents[idx]
        num_elements = num_elements[idx]

        # descent_score = descent_score[idx]
        del descent_score

        logger.debug(f"elements: {';'.join(str(_) for _ in elements)}")

        def _root_cause_score(partition: int) -> float:
            dis_f = cityblock
            data_p, data_n = self.get_derived_dataframe(
                frozenset(elements[:partition]), cuboid=cuboid,
                reduction=lambda x: x, return_complement=True,
                subset_indices=np.concatenate([indices, self.normal_indices]))
            assert len(data_p) + len(data_n) == len(indices) + len(self.normal_indices), \
                f'{len(data_n)} {len(data_p)} {len(indices)} {len(self.normal_indices)}'
            # dp = self.__deviation_score(data_p['real'].values, data_p['predict'].values)
            # dn = self.__deviation_score(data_n['real'].values, data_n['predict'].values) if len(data_n) else []
            # log_ll = np.mean(norm.pdf(dp, loc=mu, scale=sigma)) \
            #          + np.mean(norm.pdf(dn, loc=0, scale=self.option.normal_deviation_std))
            _abnormal_descent_score = np.sum(num_elements[:partition]) / np.sum(num_ele_descents[:partition])
            _normal_descent_score = 1 - np.sum(num_elements[partition:] / np.sum(num_ele_descents[partition:]))
            _ds = _normal_descent_score * _abnormal_descent_score
            succinct = partition + len(cuboid) * len(cuboid)
            _pv, _pf = np.sum(data_p.real.values), np.sum(data_p.predict.values)
            _lp = len(data_p)
            _v1, _v2 = data_p.real.values, data_n.real.values
            _v = np.concatenate([_v1, _v2])
            _f1, _f2 = data_p.predict.values, data_n.predict.values
            _f = np.concatenate([_f1, _f2])

            # TODO: fixed known issue
            reduced_data_p, _ = self.get_derived_dataframe(
                frozenset(elements[:partition]), cuboid=cuboid,
                reduction="sum", return_complement=True,
                subset_indices=np.concatenate([indices, self.normal_indices]))
            if len(reduced_data_p):
                _a1, _a2 = data_p.predict.values * (
                        reduced_data_p.real.item() / reduced_data_p.predict.item()
                ), data_n.predict.values
            else:
                assert len(data_p) == 0
                _a1 = 0
                _a2 = data_n.predict.values

            # Original:
            # _a1, _a2 = data_p.predict.values * (_pv / _pf), data_n.predict.values

            _a = np.concatenate([_a1, _a2])
            divide = lambda x, y: x / y if y > 0 else (0 if x == 0 else float('inf'))
            _ps = 1 - (divide(dis_f(_v1, _a1), len(_v1)) + divide(dis_f(_v2, _f2), len(_v2))) \
                  / (divide(dis_f(_v1, _f1), len(_v1)) + divide(dis_f(_v2, _f2), len(_v2)))
            logger.debug(
                f"partition:{partition} "
                # f"log_ll:{log_ll} "
                # f"impact: {impact_score} "
                f"succinct: {succinct} "
                f"ps: {_ps}"
            )
            # return _p * self.option.score_weight / (-succinct)
            return _ps

        partitions = np.arange(
            min(
                len(elements),
                self.option.max_num_elements_single_cluster,
                len(set(self.get_indexed_data(cuboid).index.values)) - 1
            )
        ) + 1
        if len(partitions) <= 0:
            return elements, float('-inf')
        rc_scores = np.asarray(list(map(_root_cause_score, partitions)))
        idx = np.argsort(rc_scores)[::-1]
        partitions = partitions[idx]
        rc_scores = rc_scores[idx]

        score = rc_scores[0]
        rc = elements[:partitions[0].item()]
        logger.debug(f"cuboid {cuboid} gives root cause {AC.batch_to_string(rc)} with score {score}")
        return rc.tolist(), score

    def _locate_in_cluster(self, indices: np.ndarray):
        """
        :param indices:  indices of leaf nodes in this cluster
        :return: None
        """
        mu = np.mean(self.leaf_deviation_score[indices])
        sigma = np.maximum(np.std(self.leaf_deviation_score[indices]), 1e-4)
        logger.debug(f"locate in cluster: {mu}(+-{sigma})")
        max_cuboid_layer = len(self.attribute_names)
        ret_lists = []
        for cuboid_layer in np.arange(max_cuboid_layer) + 1:
            layer_ret_lists = list(map(
                lambda x, _i=indices, _mu=mu, _sigma=sigma: self._locate_in_cuboid(x, indices=_i, mu=_mu, sigma=_sigma),
                combinations(self.attribute_names, cuboid_layer)
            ))
            ret_lists.extend([
                {
                    'rc': x[0], 'score': x[1], 'n_ele': len(x[0]), 'layer': cuboid_layer,
                    'rank': x[1] * self.option.score_weight - len(x[0]) * cuboid_layer
                } for x in layer_ret_lists
            ])
            if len(list(filter(lambda x: x['score'] > self.option.ps_upper_bound, ret_lists))):
                break
        ret_lists = list(sorted(
            ret_lists,
            key=lambda x: x['rank'],
            reverse=True)
        )
        if ret_lists:
            ret = ret_lists[0]['rc']
            logger.debug(
                f"find root cause: {AC.batch_to_string(ret)}, rank: {ret_lists[0]['rank']}, score: {ret_lists[0]['score']}")
            self._root_cause.append(frozenset(ret))
        else:
            logger.info("failed to find root cause")

    def locate_root_cause(self):
        if not self.cluster_list:
            logger.info("We do not have abnormal points")
            return
        if self.option.score_weight == 'auto':
            # TODO: Using the revised score formula
            # see: https://github.com/NetManAIOps/Squeeze/issues/6
            g_cluster = np.log(len(self.cluster_list) + 1) / len(self.cluster_list)
            num_attr = sum([len(attrs) for attrs in self.attribute_values])
            g_attribute = num_attr / np.log(num_attr + 1)
            g_coverage = -np.log(sum([len(cluster) for cluster in self.cluster_list]) / len(self.data_list[0]))
            self.option.score_weight = g_cluster * g_attribute * g_coverage

            # Original:
            #self.option.score_weight = - np.log(
            #   len(self.cluster_list) *
            #   sum(len(_) for _ in self.cluster_list) / len(self._f)) / np.log(
            #   sum(len(_) for _ in self.attribute_values)) * sum(len(_) for _ in self.attribute_values)
            #self.option.score_weight = len(self.cluster_list) * \
            #                           (np.log(sum(len(_) for _ in self.cluster_list)) + np.sum(
            #                               [np.log(len(_)) for _ in self.attribute_values]) - np.log(
            #                               len(self.cluster_list)) - np.log(len(self.leaf_deviation_score))) \
            #                           / np.log(np.mean([len(_) for _ in self.attribute_values])) * 10
            logger.debug(f"auto score weight: {self.option.score_weight}")
        for indices in self.cluster_list:
            self._locate_in_cluster(indices)

    @property
    @lru_cache()
    def leaf_deviation_score(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_scores = self.__deviation_score(self._v, self._f)
        assert np.shape(deviation_scores) == np.shape(self._v) == np.shape(self._f)
        assert np.sum(np.isnan(deviation_scores)) == 0, \
            f"there are nan in deviation score {np.where(np.isnan(deviation_scores))}"
        assert np.sum(~np.isfinite(deviation_scores)) == 0, \
            f"there are infinity in deviation score {np.where(~np.isfinite(deviation_scores))}"
        logger.debug(f"anomaly ratio ranges in [{np.min(deviation_scores)}, {np.max(deviation_scores)}]")
        return deviation_scores

    def get_derived_dataframe(self, ac_set: Union[FrozenSet[AC], None], cuboid: Tuple[str] = None,
                              reduction=None, return_complement=False, subset_indices=None):
        subset = np.zeros(len(self.data_list[0]), dtype=np.bool)
        if subset_indices is not None:
            subset[subset_indices] = True
        else:
            subset[:] = True

        if reduction == "sum":
            reduce = lambda x, _axis=0: np.sum(x, axis=_axis, keepdims=True)
        else:
            reduce = lambda x: x

        if ac_set is None:
            idx = np.ones(shape=(len(self.data_list[0]),), dtype=np.bool)
        else:
            idx = AC.batch_index_dataframe(ac_set, self.get_indexed_data(cuboid))

        def _get_ret(_data_list):
            if len(_data_list[0]) == 0:
                return pd.DataFrame(data=[], columns=['real', 'predict'])
            _values = self.op(*[reduce(_data[["real", "predict"]].values) for _data in _data_list])
            if np.size(_values) == 0:
                _values = []
            if reduction == 'sum':
                _ret = pd.DataFrame(data=_values, columns=['real', 'predict'])
            else:
                _ret = _data_list[0].copy(deep=True)
                _ret[['real', 'predict']] = _values
            return _ret

        data_list = list(_[idx & subset] for _ in self.data_list)
        if not return_complement:
            return _get_ret(data_list)
        complement_data_list = list(_[(~idx) & subset] for _ in self.data_list)
        return _get_ret(data_list), _get_ret(complement_data_list)

    @staticmethod
    def __deviation_score(v, f):
        n = 1
        with np.errstate(divide='ignore'):
            ret = n * (f - v) / (n * f + v)
            # ret = np.log(np.maximum(v, 1e-10)) - np.log(np.maximum(f, 1e-10))
            # ret = (2 * sigmoid(1 - v / f) - 1)
            # k = np.log(np.maximum(v, 1e-100)) - np.log(np.maximum(f, 1e-100))
            # ret = (1 - k) / (1 + k)
        ret[np.isnan(ret)] = 0.
        return ret
