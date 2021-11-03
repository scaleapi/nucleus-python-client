{{ objname | escape | underline }}

.. automodule:: {{ fullname }}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
