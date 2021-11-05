{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ name }}
   :members:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in methods if item != '__init__' %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
