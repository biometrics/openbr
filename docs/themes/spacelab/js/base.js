
/* Highlight */
$( document ).ready(function() {
    hljs.initHighlightingOnLoad();
    $('table').addClass('table table-striped table-hover');
});


$('body').scrollspy({
    target: '#bs-sidebar',
});

$('[data-spy="scroll"]').each(function()
{
    $(this).scrollspy('refresh');
});

$('#myScrollspy').on('activate.bs.scrollspy', function () {
    $(this).parent().children('ul.scroll_toggle').toggle(50)
})

/* Prevent disabled links from causing a page reload */
$("li.disabled a").click(function() {
    event.preventDefault();
});
