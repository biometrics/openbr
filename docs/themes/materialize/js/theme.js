$(document).ready(function(){
    $('.toc-wrapper').pushpin({ top: $('nav').height() });

    $('.scrollspy').scrollSpy();
    $('.button-collapse').sideNav({'edge': 'left'});
})

$('.button-collapse').sideNav({
      menuWidth: 300, // Default is 240
      edge: 'right', // Choose the horizontal origin
      closeOnClick: true // Closes side-nav on <a> clicks, useful for Angular/Meteor
    }
  );
$('.collapsible').collapsible();
