# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import re
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select

from airflow.api_fastapi.common.parameters import (
    FilterParam,
    SortParam,
    _PrefixPatternParam,
    _PrefixSearchParam,
    _SearchParam,
    _TaskDisplayNamePrefixPatternParam,
    filter_param_factory,
)
from airflow.models import DagModel, DagRun, Log
from airflow.models.taskinstance import TaskInstance


class TestFilterParam:
    def test_filter_param_factory_description(self):
        app = FastAPI()  # Create a FastAPI app to test OpenAPI generation
        expected_descriptions = {
            "dag_id": "Filter by Dag ID Description",
            "task_id": "Filter by Task ID Description",
            "map_index": None,  # No description for map_index
            "run_id": "Filter by Run ID Description",
        }

        @app.get("/test")
        def test_route(
            dag_id: Annotated[
                FilterParam[str | None],
                Depends(
                    filter_param_factory(Log.dag_id, str | None, description="Filter by Dag ID Description")
                ),
            ],
            task_id: Annotated[
                FilterParam[str | None],
                Depends(
                    filter_param_factory(Log.task_id, str | None, description="Filter by Task ID Description")
                ),
            ],
            map_index: Annotated[
                FilterParam[int | None],
                Depends(filter_param_factory(Log.map_index, int | None)),
            ],
            run_id: Annotated[
                FilterParam[str | None],
                Depends(
                    filter_param_factory(
                        DagRun.run_id, str | None, description="Filter by Run ID Description"
                    )
                ),
            ],
        ):
            return {"message": "test"}

        # Get the OpenAPI spec
        openapi_spec = app.openapi()

        # Check if the description is in the parameters
        parameters = openapi_spec["paths"]["/test"]["get"]["parameters"]
        for param_name, expected_description in expected_descriptions.items():
            param = next((p for p in parameters if p.get("name") == param_name), None)
            assert param is not None, f"{param_name} parameter not found in OpenAPI"

            if expected_description is None:
                assert "description" not in param, (
                    f"Description should not be present in {param_name} parameter"
                )
            else:
                assert "description" in param, f"Description not found in {param_name} parameter"
                assert param["description"] == expected_description, (
                    f"Expected description '{expected_description}', got '{param['description']}'"
                )


class TestSortParam:
    def test_sort_param_max_number_of_filers(self):
        param = SortParam([], None, None)
        n_filters = param.MAX_SORT_PARAMS + 1
        param.value = [f"filter_{i}" for i in range(n_filters)]

        with pytest.raises(
            HTTPException,
            match=re.escape(
                f"400: Ordering with more than {param.MAX_SORT_PARAMS} parameters is not allowed. Provided: {param.value}"
            ),
        ):
            param.to_orm(None)


def _compile(statement):
    return str(statement.compile(compile_kwargs={"literal_binds": True})).lower()


def _has_ilike(sql: str, term: str) -> bool:
    """Return True if ``sql`` contains an ILIKE (or dialect-equivalent) match for ``%term%``."""
    return f"'%{term.lower()}%'" in sql


class TestSearchParam:
    """Substring search (``ILIKE '%term%'``) — full-match, case-insensitive."""

    def test_to_orm_single_value(self):
        param = _SearchParam(DagModel.dag_id).set_value("example_bash")
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert _has_ilike(sql, "example_bash")

    def test_to_orm_none_value_is_noop(self):
        """A ``None`` value with ``skip_none`` leaves the statement unchanged."""
        param = _SearchParam(DagModel.dag_id).set_value(None)
        statement = select(DagModel)
        result = param.to_orm(statement)
        assert result is statement

    def test_to_orm_tilde_alias_matches_all(self):
        """``~`` is aliased to ``%`` so the ILIKE expression matches all rows."""
        param = _SearchParam(DagModel.dag_id)
        aliased = param.transform_aliases("~")
        param.set_value(aliased)
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert _has_ilike(sql, "%")

    def test_to_orm_multiple_values_or(self):
        """Test search with multiple terms using the pipe | operator."""
        param = _SearchParam(DagModel.dag_id).set_value("example_bash | example_python")
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert "or" in sql
        assert _has_ilike(sql, "example_bash")
        assert _has_ilike(sql, "example_python")

    def test_to_orm_pipe_with_trailing_pipe(self):
        """Test that a trailing pipe is ignored and only the valid term is searched."""
        param = _SearchParam(DagModel.dag_id).set_value("example_bash|")
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert _has_ilike(sql, "example_bash")
        assert " or " not in sql

    def test_to_orm_pipe_with_leading_pipe(self):
        """Test that a leading pipe is ignored and only the valid term is searched."""
        param = _SearchParam(DagModel.dag_id).set_value("|example_bash")
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert _has_ilike(sql, "example_bash")
        assert " or " not in sql


class TestPrefixSearchParam:
    """Prefix search using range comparison (``attribute >= term AND < upper``)."""

    def test_to_orm_single_value(self):
        param = _PrefixSearchParam(DagModel.dag_id).set_value("example_bash")
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert "dag_id >= 'example_bash'" in sql
        assert "dag_id < 'example_basi'" in sql

    def test_prefix_range_upper_boundary(self):
        """Prefix range correctly increments the last character."""
        assert _PrefixPatternParam._prefix_range_upper("xy") == "xz"
        assert _PrefixPatternParam._prefix_range_upper("test_dag") == "test_dah"
        assert _PrefixPatternParam._prefix_range_upper("") is None

    def test_to_orm_empty_value_matches_all(self):
        """An empty value matches all non-null rows (used via the ``~`` alias)."""
        param = _PrefixSearchParam(DagModel.dag_id).set_value("")
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert "is not null" in sql

    def test_to_orm_tilde_alias_matches_all(self):
        param = _PrefixSearchParam(DagModel.dag_id)
        aliased = param.transform_aliases("~")
        param.set_value(aliased)
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert "is not null" in sql

    def test_to_orm_multiple_values_or(self):
        param = _PrefixSearchParam(DagModel.dag_id).set_value("example_bash | example_python")
        statement = select(DagModel)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert "or" in sql
        assert "example_bash" in sql
        assert "example_python" in sql

    def test_to_orm_none_value_is_noop(self):
        param = _PrefixSearchParam(DagModel.dag_id).set_value(None)
        statement = select(DagModel)
        result = param.to_orm(statement)
        assert result is statement


class TestTaskDisplayNamePrefixPatternParam:
    """Prefix filter splits on NULL override so ``task_id`` can use indexes."""

    def test_to_orm_uses_task_id_when_override_null(self):
        param = _TaskDisplayNamePrefixPatternParam().set_value("test_task_hello")
        statement = select(TaskInstance)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert "task_display_name is null" in sql
        assert "task_id >=" in sql
        assert "task_id <" in sql
        assert "task_display_name is not null" in sql

    def test_to_orm_empty_matches_all(self):
        param = _TaskDisplayNamePrefixPatternParam().set_value("")
        statement = select(TaskInstance)
        statement = param.to_orm(statement)

        sql = _compile(statement)
        assert "true" in sql or "1 = 1" in sql
