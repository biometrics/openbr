Members | Type | Description
--- | --- | ---
<a class="table-anchor" id="timeinvariantalias"></a> timeInvariantAlias | TimeInvariantWrapperTransform | A special case helper class that ensures usually thread safe calls like [project](../transform/functions.md#project-1) can be called non-thread safe objects. In a very simplified description, this calls [projectUpdate](functions.md#projectupdate-1) instead of [project](functions.md#project-1)
