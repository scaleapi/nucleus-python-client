{% set exclude_classes = ['ModelRun',
                          'MultiCategoryAnnotation'] %}

{{ objname | escape | underline }}

.. automodule:: {{ fullname }}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in classes if item not in exclude_classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
